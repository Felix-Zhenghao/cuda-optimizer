# 9.7.13.10-11. match.sync, activemask

match.sync broadcasts and compares values across warp threads: match.any returns mask of threads with same value, match.all checks unanimity. activemask queries which threads are currently active in the warp, returning a 32-bit bitmask indexed by lane id.
