# 9.7.13.1. bar, barrier

CTA-level barrier synchronization using 16 named barriers. Supports sync (wait for all), arrive (signal without waiting), and red (reduction with popc/and/or). Thread counts must be warp-size multiples. Enables producer/consumer patterns with mixed arrive/sync usage.
