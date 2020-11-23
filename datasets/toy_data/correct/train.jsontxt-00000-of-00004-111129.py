

@anonymous_csrf
@ratelimit(field='username', method='POST', rate='5/m')
def login(request):
    if ('next' in request.GET):
        try:
            resolve(request.GET['next'])
        except Http404:
            q = request.GET.copy()
            q.update({
                'next': '/',
            })
            request.GET = q
    kwargs = {
        'template_name': 'users/login.html',
        'authentication_form': forms.CaptchaAuthenticationForm,
    }
    if settings.USE_BROWSERID:
        kwargs['template_name'] = 'users/browserid_login.html'
    if (request.method == 'POST'):
        kwargs['authentication_form'] = partial(kwargs['authentication_form'], request)
    return auth_views.login(request, **kwargs)
