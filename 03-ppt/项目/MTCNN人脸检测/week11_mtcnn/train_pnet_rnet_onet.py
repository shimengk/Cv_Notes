import threading

from week11_mtcnn.trainer import Trainer
from week11_mtcnn.mtcnn import PNet, RNet, ONet


def train_model(model_id):
    if model_id == 1:
        Trainer(PNet(), r"params/p_net.pt", r"C:\lfw_out_add4\12", batch_size=1024).train()
    elif model_id == 2:
        Trainer(RNet(), r"params/r_net.pt", r"C:\lfw_out_add4\24", batch_size=2048).train()
    elif model_id == 3:
        Trainer(ONet(), r"params/o_net.pt", r"C:\lfw_out_add4\48", batch_size=512).train()


# 创建三个训练器的线程
thread1 = threading.Thread(target=train_model, args=(1,))
thread2 = threading.Thread(target=train_model, args=(2,))
thread3 = threading.Thread(target=train_model, args=(3,))

# 启动线程
thread1.start()
thread2.start()
thread3.start()

# 等待所有线程完成
thread1.join()
thread2.join()
thread3.join()