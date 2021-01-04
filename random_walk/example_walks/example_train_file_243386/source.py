

def test_get_collection_custom_fields(self):
    fields = 'uuid,extra'
    for i in range(3):
        obj_utils.create_test_port(self.context, node_id=self.node.id, uuid=uuidutils.generate_uuid(), address=('52:54:00:cf:2d:3%s' % i))
    data = self.get_json(('/ports?fields=%s' % fields), headers={
        api_base.Version.string: str(api_v1.MAX_VER),
    })
    self.assertEqual(3, len(data['ports']))
    for port in data['ports']:
        self.assertItemsEqual(['uuid', 'extra', 'links'], port)
