def compile_message(self, message_template, context):
        try:
            msg = six.text_type(message_template).format(**context)
        except HoleException as e:
            raise LoggerError(
                "Cannot find %s context field. Choices are: %s" % (
                    str(e), ', '.join(context.keys())))
        return msg
