/*
 * flipcast_sizes: make the planner believe pg_class sizes.
 *
 * the planner normally ignores relpages/reltuples as absolute values. it
 * asks the storage manager how many blocks a relation has right now and only
 * uses reltuples/relpages as a density (see table_block_relation_estimate_size
 * in tableam.c). that is the right call for a real database, but it means an
 * empty copy of a schema plans as if every table had zero rows, no matter
 * what pg_restore_relation_stats() wrote into the catalog.
 *
 * this module hooks get_relation_info and overwrites the size estimates with
 * the catalog values, for tables and their indexes. column statistics are not
 * touched; the planner already reads those straight from pg_statistic.
 *
 * load it per session with LOAD 'flipcast_sizes' (or session_preload_libraries)
 * on a scratch server only. never on a server that holds real data.
 */
#include "postgres.h"

#include <math.h>

#include "access/htup_details.h"
#include "catalog/pg_am_d.h"
#include "catalog/pg_class.h"
#include "fmgr.h"
#include "nodes/pathnodes.h"
#include "optimizer/plancat.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/syscache.h"

PG_MODULE_MAGIC_EXT(.name = "flipcast_sizes", .version = "0.3");

static bool enabled = true;
static bool installed = false;
static get_relation_info_hook_type prev_get_relation_info = NULL;

typedef struct CatalogSize
{
	char		relkind;
	BlockNumber pages;
	double		tuples;
	BlockNumber allvisible;
} CatalogSize;

static bool
catalog_size(Oid relid, CatalogSize *out)
{
	HeapTuple	tup;
	Form_pg_class form;

	tup = SearchSysCache1(RELOID, ObjectIdGetDatum(relid));
	if (!HeapTupleIsValid(tup))
		return false;
	form = (Form_pg_class) GETSTRUCT(tup);
	out->relkind = form->relkind;
	out->pages = (BlockNumber) form->relpages;
	out->tuples = (double) form->reltuples;
	out->allvisible = (BlockNumber) form->relallvisible;
	ReleaseSysCache(tup);

	/* reltuples = -1 means nobody ever analyzed or injected anything */
	return out->tuples >= 0;
}

/*
 * guess the btree height from size alone. the real value lives in the
 * metapage, which is empty here. height is about log_fanout(leaf pages).
 *
 * tuples per leaf page overstates the fanout of inner pages: leaves get
 * deduplicated, inner pages don't, and inner pages sit around 70% full. so
 * the leaf density is capped. 350 matched real indexes on int, bigint and
 * timestamptz keys in testing, and one level off only moves the descent cost
 * by 50 * cpu_operator_cost.
 */
#define MAX_INNER_FANOUT 350.0

static int
guess_btree_height(BlockNumber pages, double tuples)
{
	double		leaves = (double) pages - 1.0;	/* minus the metapage */
	double		fanout;

	if (leaves <= 1.0 || tuples <= 0)
		return 0;
	fanout = tuples / leaves;
	if (fanout > MAX_INNER_FANOUT)
		fanout = MAX_INNER_FANOUT;
	if (fanout < 2.0)
		fanout = 2.0;
	return (int) ceil(log(leaves) / log(fanout));
}

static void
flipcast_get_relation_info(PlannerInfo *root, Oid relationObjectId,
						   bool inhparent, RelOptInfo *rel)
{
	CatalogSize heap;
	ListCell   *lc;

	if (prev_get_relation_info)
		prev_get_relation_info(root, relationObjectId, inhparent, rel);

	/* partitioned parents get their size from the children, leave them be */
	if (!enabled || inhparent)
		return;
	if (!catalog_size(relationObjectId, &heap))
		return;
	if (heap.relkind != RELKIND_RELATION && heap.relkind != RELKIND_MATVIEW)
		return;

	rel->pages = heap.pages;
	rel->tuples = heap.tuples;
	if (heap.pages > 0)
		rel->allvisfrac = Min(1.0, (double) heap.allvisible / (double) heap.pages);
	else
		rel->allvisfrac = 0.0;

	foreach(lc, rel->indexlist)
	{
		IndexOptInfo *info = (IndexOptInfo *) lfirst(lc);
		CatalogSize idx;

		if (!catalog_size(info->indexoid, &idx))
			continue;

		info->pages = idx.pages;
		/* same rule as get_relation_info: full indexes match the heap */
		if (info->indpred == NIL)
			info->tuples = rel->tuples;
		else
			info->tuples = Min(idx.tuples, rel->tuples);

		if (info->relam == BTREE_AM_OID)
			info->tree_height = guess_btree_height(info->pages, info->tuples);
	}
}

PG_FUNCTION_INFO_V1(flipcast_sizes_active);

/* lets a client check that the hook is installed in its own backend */
Datum
flipcast_sizes_active(PG_FUNCTION_ARGS)
{
	PG_RETURN_BOOL(enabled && installed);
}

void
_PG_init(void)
{
	DefineCustomBoolVariable("flipcast_sizes.enabled",
							 "Use pg_class relpages and reltuples as the planner's relation size.",
							 NULL,
							 &enabled,
							 true,
							 PGC_USERSET,
							 0,
							 NULL, NULL, NULL);
	MarkGUCPrefixReserved("flipcast_sizes");

	prev_get_relation_info = get_relation_info_hook;
	get_relation_info_hook = flipcast_get_relation_info;
	installed = true;
}
