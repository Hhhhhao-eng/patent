import os
import django
from django.urls import get_resolver

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sulab.settings')
django.setup()


def print_urls(urlpatterns=None, prefix=''):
    if urlpatterns is None:
        urlpatterns = get_resolver().url_patterns
    for pattern in urlpatterns:
        if hasattr(pattern, 'url_patterns'):  # 包含其他URLconf
            print_urls(pattern.url_patterns, prefix + str(pattern.pattern))
        else:
            print(f"{prefix}{pattern.pattern} -> {pattern.callback.__module__}.{pattern.callback.__name__}")


print("Registered URLs:")
print_urls()
