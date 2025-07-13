MBROLA DLL build version 3.3 by Thiekus (revision 2)
---------------------------------------
Here is precompiled mbrola.dll and mbrola.exe by either Microsoft Visual C++ 6.0 or Visual C++ 2015 using updated open-source mbrola
source code. This build is API compatible by original library, only lacks nagging license dialog that spawn every time you load
mbrola.dll library and has more recent change fixes. Also, this build contains 64-bit binary build that lacks in any original binary distribution.

In the fact, the mbrola source code is support Visual C++ compiler for build on Windows with only slightly changes, even by MSVC6!

How to use?
-----------
Depends your application, if you compile into 32bit program, then use mbrola.dll from Win32\Release or if you compile into 64bit
program, use mbrola.dll from Win64\Release. The debug binary also included if you desire to use that. Note that MSVC6 build is
shipped in VC6 dir only for compatibilty reason, if you needed to run your program into Windows older than Windows XP and use MSVC2015
built binary instead if possible.

Note: This library is subject of GNU AGPL 3.0 license as original mbrola source code.

Fork repo: https://github.com/thiekus/MBROLA
