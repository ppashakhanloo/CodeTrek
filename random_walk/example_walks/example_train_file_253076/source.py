

def setUp(self):
    self.srid = Location.geom._field.srid
    self.view = generics.GeoListView.as_view(queryset=Location.objects.all())
    records = [{
        'name': 'Banff',
        'coordinates': [(- 115.554), 51.179],
    }, {
        'name': 'Jasper',
        'coordinates': [(- 118.081), 52.875],
    }]
    for record in records:
        obj = Location.add_buffer(record.pop('coordinates'), 0.5, **record)
    self.qs = Location.objects.all()
