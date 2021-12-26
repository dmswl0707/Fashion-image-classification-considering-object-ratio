import os

folder_path = '/Users/choieunji/Desktop/dataset/Train/pair1/crop_tee'

folder_list = os.listdir(folder_path)


for fname1 in folder_list:
    #해당 test 폴더(1,2,3,4) 위치 설정
    current_folder = folder_path + "/" + fname1
    #각 test폴더(1,2,3,4) 안의 파일명 받아오기
    filelist = os.listdir(current_folder)

    print("현재 폴더명 : ", fname1)
    #각 폴더명의 파일리스트를 다시 for문을 통해 반복
    for fname2 in filelist:
        #os.rename(a, b) : a를 b로 이름 변경. b는 저장될 위치도 지정하는 것이므로 같은 폴더에하려면 폴더명 지정
        print(fname2+"를 result"+str(i)+".jpg로 파일명을 변경합니다.")
        os.rename(current_folder+"/"+fname2, current_folder+"/"+"이미지"+str(i)+".jpg")
        i = i+1

'''
def changeName(path, cName):
    i = 1
    for filename in os.listdir(path):
        print(path + filename, '=>', path + str(cName) + str(i) + '.jpg')
        os.rename(path + filename, path + str(cName) + str(i) + '.jpg')
        i += 1


changeName('/Users/aaron/Desktop/testFolder/', 'test')
'''

