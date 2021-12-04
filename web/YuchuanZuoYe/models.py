from django.db import models

# Create your models here.
class Classfication(models.Model):
    """用户预测数据的主题包括：时间、文本信息、等"""
    text = models.CharField(max_length = 200)
    date_added = models.DateTimeField(auto_now_add = True)
    #owner = models.ForeignKey(User,on_delete=models.CASCADE)
        
    def __str__(self):
        """返回模型的字符串表示"""
        return self.text
	

class UploadFile(models.Model):
    userid = models.CharField(max_length = 30)
    file = models.FileField(upload_to = './upload/')
    date = models.DateTimeField(auto_now_add=True)
