from hypothesis import given
from hypothesis import strategies as st

from flipcast import pgarray, sqltext

# --- array literals


def test_parse_plain_and_quoted():
    assert pgarray.parse("{1,2,3}") == ["1", "2", "3"]
    assert pgarray.parse('{"a b","c,d",e}') == ["a b", "c,d", "e"]
    assert pgarray.parse('{"say \\"hi\\"","back\\\\slash"}') == ['say "hi"', "back\\slash"]
    assert pgarray.parse("{}") == []
    assert pgarray.parse("{NULL,x}") == [None, "x"]
    assert pgarray.parse('{"NULL"}') == ["NULL"]  # quoted NULL is a string


def test_parse_timestamps_as_pg_prints_them():
    text = '{"2026-01-01 00:00:09+00","2026-01-01 02:34:02+00"}'
    assert pgarray.parse(text) == ["2026-01-01 00:00:09+00", "2026-01-01 02:34:02+00"]


def test_parse_nested_elements_are_kept_whole():
    # stats on an array column hold arrays as elements
    assert pgarray.parse("{{1,2},{3,4}}") == ["{1,2}", "{3,4}"]
    assert pgarray.parse('{"{1,2}","{3}"}') == ["{1,2}", "{3}"]


def test_parse_lower_bound_prefix():
    assert pgarray.parse("[0:1]={a,b}") == ["a", "b"]


@given(
    st.lists(
        st.one_of(
            st.none(),
            st.text(
                alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00")
            ),
        )
    )
)
def test_roundtrip(elems):
    assert pgarray.parse(pgarray.to_literal(elems)) == elems


# --- sql lexer


def test_split_statements_respects_quotes_and_bodies():
    script = """
    create function f() returns int language sql as $$ select 1; $$;
    -- a comment; with a semicolon
    insert into t values ('a;b', E'it\\'s;');
    /* block; comment */ select 2
    """
    stmts = sqltext.split_statements(script)
    assert len(stmts) == 3
    assert stmts[0].startswith("create function")
    assert "'a;b'" in stmts[1]
    assert stmts[2].endswith("select 2")


def test_split_drops_empty_and_comment_only():
    assert sqltext.split_statements(";;  -- nothing\n;") == []


def test_max_param_ignores_strings_and_dollar_quotes():
    assert sqltext.max_param("select $1, '$5', $2") == 2
    assert sqltext.max_param("select $tag$ $9 $tag$, $3") == 3
    assert sqltext.max_param("select 1") == 0


def test_rewrite_clock():
    out = sqltext.rewrite_clock(
        "select * from t where a > now() - interval '1 day' and b = 'now()'"
        " and c < pg_catalog.now() and d = current_date and e > current_timestamp(3)",
        tstz="2027-01-01 00:00:00+00:00",
        ts="2027-01-01 00:00:00",
        d="2027-01-01",
    )
    assert "now()" not in out.replace("'now()'", "")
    assert "b = 'now()'" in out  # inside a string, untouched
    assert "pg_catalog" not in out
    assert "('2027-01-01'::date)" in out
    assert out.count("::timestamptz") == 3


def test_rewrite_clock_leaves_columns_named_now():
    sql = "select now from t"
    assert sqltext.rewrite_clock(sql, tstz="x", ts="x", d="x") == sql


def test_fix_typed_params():
    assert sqltext.fix_typed_params("select now() - interval $1") == "select now() - $1::interval"
    assert sqltext.fix_typed_params("select date $2 + 1") == "select $2::date + 1"
    # a column called "date" compared to a param stays alone
    assert sqltext.fix_typed_params("select * from t where date = $1") == (
        "select * from t where date = $1"
    )


def test_fingerprint_matches_pg_stat_statements_folding():
    a = "SELECT id FROM events\n WHERE account_id = $1 ORDER BY created_at DESC LIMIT 20"
    b = "select id from events where account_id = $1 order by created_at desc limit $2"
    assert sqltext.fingerprint(a) == sqltext.fingerprint(b)
    assert sqltext.fingerprint("select 1") != sqltext.fingerprint("select 1 from t")
