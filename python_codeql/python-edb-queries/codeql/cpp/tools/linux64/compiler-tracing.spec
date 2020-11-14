**/configure:
  order compiler
  trace no
**/do-prebuild:
  order compiler
  trace no
**/collect2:
  invoke ${config_dir}/extractor
  prepend --linker
  prepend --semmle-linker-executable
  prepend ${compiler}
**/ld*:
**/*-ld*:
**/lld*:
  invoke ${config_dir}/extractor
  prepend --linker
  prepend --semmle-linker-executable
  prepend ${compiler}
**/armlink:
  invoke ${config_dir}/extractor
  prepend --arm-linker
  prepend --semmle-linker-executable
  prepend ${compiler}
**/*clang*:
**/*cc*:
**/*++*:
**/icpc:
  invoke ${config_dir}/extractor
  order compiler, extractor
  prepend --mimic
  prepend ${compiler}
