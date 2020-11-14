**/cl.exe:
**/clang-cl.exe:
  invoke ${config_dir}/extractor.exe
  order compiler, extractor
  prepend --mimic
  prepend "${compiler}"
**/link.exe:
**/lld-link.exe:
  invoke ${config_dir}/extractor.exe
  prepend --ms-linker
  prepend --semmle-linker-executable
  prepend "${compiler}"
**/collect2.exe:
  invoke ${config_dir}/extractor.exe
  prepend --linker
  prepend --semmle-linker-executable
  prepend "${compiler}"
**/ld*.exe:
**/*-ld*.exe:
  invoke ${config_dir}/extractor.exe
  prepend --linker
  prepend --semmle-linker-executable
  prepend "${compiler}"
**/clang*.exe:
**/gcc*.exe:
**/g++*.exe:
**/armcc.exe:
**/*-clang.exe:
**/*-gcc.exe:
**/*-g++.exe:
**/c51.exe:
**/cx51.exe:
  invoke ${config_dir}/extractor.exe
  order compiler, extractor
  prepend --mimic
  prepend "${compiler}"
