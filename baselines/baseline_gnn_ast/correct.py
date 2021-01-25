

@classmethod
def schema_from_record(cls, record):
    from bigquery.schema_builder import schema_from_record
    return schema_from_record(record)
