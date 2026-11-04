\echo Use "CREATE EXTENSION flipcast_sizes" to load this file. \quit

-- true when the hook is loaded in the calling backend. creating the extension
-- does not load the library by itself, a session still needs LOAD or
-- session_preload_libraries.
CREATE FUNCTION flipcast_sizes_active() RETURNS boolean
AS 'MODULE_PATHNAME', 'flipcast_sizes_active'
LANGUAGE C STRICT VOLATILE;
