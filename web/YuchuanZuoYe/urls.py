"""定义YuchuanZuoYe的URL模式"""
from django.conf.urls import url
from django.conf.urls.static import static
from django.conf import settings
from . import views

from django.contrib import admin
from django.urls import path

app_name ='myurl'
urlpatterns = [
    #主页
    url(r'^$',views.index, name = 'index'),
    #显示结果的主页
    url(r'^classfication1/$',views.classfication1, name = 'classfication1'),
    #可视化显示结果的主页
    url(r'^classfication2/$',views.classfication2, name = 'classfication2'),
    #重定向upload
]
