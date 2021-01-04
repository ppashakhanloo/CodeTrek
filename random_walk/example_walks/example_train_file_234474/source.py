

def RenderAjax(self, request, response):
    'Return the count on unseen notifications.'
    response = super(NotificationCount, self).RenderAjax(request, response)
    number = 0
    try:
        user_fd = aff4.FACTORY.Open(aff4.ROOT_URN.Add('users').Add(request.user), token=request.token)
        notifications = user_fd.Get(user_fd.Schema.PENDING_NOTIFICATIONS)
        if notifications:
            number = len(notifications)
    except IOError:
        pass
    return renderers.JsonResponse(dict(number=number))
