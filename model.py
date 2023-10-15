import torch
from PIL import Image
import easyocr
import cv2


class Gvozdomet:
    def init(self):
        self.img_path = None


    def connect_model(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='train_num_yolov5_weights.pt', force_reload=True)
        self.model.conf = 0.5 #нижняя планка вероятности для предсказания
        self.reader = easyocr.Reader(['en'], gpu=False)


    def get_number(self, pth):

        # self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='/content/drive/MyDrive/num_weights/train_num_yolov5_weights.pt', force_reload=True)
        # self.model.conf = 0.5 #нижняя планка вероятности для предсказания
        self.arr = [pth]
        img = Image.open(pth)
        img = img.resize((640, 640))
        self.results = self.model(img)

        return self.detect_train_number(pth)


    def detect_train_number(self, pth):

        img = cv2.resize(cv2.imread(pth), (640, 640))
        if self.results.xyxy[0].shape[0] > 0:
            mas = self.results.xyxy[0][0].cpu().numpy().tolist()[:-2]
            for i in range(len(mas)):
                mas[i] = int(round(mas[i], 0))


            img_rec = img[mas[1]:mas[3], mas[0]:mas[2]]


            img_rec = cv2.cvtColor(img_rec, cv2.COLOR_BGR2GRAY)


            detections = self.reader.readtext(img_rec, detail=0)
            if len(detections) > 0:
                itog_nm = ""
                num = detections[0]
                for i in num:
                    if i in "0123456789":
                        itog_nm += i
                    elif i in "\|/":
                        itog_nm += "1"
                if len(itog_nm) > 0:
                    self.arr.append(1) #добавили 1 в type
                    self.arr.append(int(itog_nm)) # добавили номер


                    if len(itog_nm) == 8:


                        mnog = "21212121"
                        sm = ""
                        for i in range(len(itog_nm) - 1):
                            sm += str(int(itog_nm[i]) * int(mnog[i]))

                        itog_sm = 0
                        for i in sm:
                            itog_sm += int(i)


                        if 10 - (itog_sm % 10) == int(itog_nm[-1]):
                            self.arr.append(1) #добавили is_correct 1 если прошло
                        else:
                            self.arr.append(0)#добавили is_correct 0 если не прошло

                    else:
                        self.arr.append(0) #добавили is_correct 0 если длина != 8
                else:
                    self.arr.append(0)
                    self.arr.append(94012576)
                    self.arr.append(0)
        else: # заливаем форму при отсутствии
            self.arr.append(0)
            self.arr.append(94012576)
            self.arr.append(0)

        return self.arr