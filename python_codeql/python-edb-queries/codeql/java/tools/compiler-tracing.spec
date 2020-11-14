jvm_prepend_arg -javaagent:${config_dir}/codeql-java-agent.jar=ignore-project,java
jvm_prepend_arg -Xbootclasspath/a:${config_dir}/codeql-java-agent.jar