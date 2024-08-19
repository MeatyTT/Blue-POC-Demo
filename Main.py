import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import os
import math
import multiprocessing
# from image_enhancement.run import *
# Thêm style trực tiếp
# st.markdown("""
# <style>
# .stFileUploader {
#     margin-bottom: 10px;
#     height: 300px;
# }
# .caption {
#     font-size: 24px;
#     text-align: center;
# }
# </style>
# """, unsafe_allow_html=True)
def is_parallel(line1, line2, tolerance=2):
  # Tính vector hướng của mỗi đường thẳng
  dx1 = line1[2] - line1[0]
  dy1 = line1[3] - line1[1]
  dx2 = line2[2] - line2[0]
  dy2 = line2[3] - line2[1]
  angle1=np.degrees(math.atan2(dy1, dx1))
  angle2=np.degrees(math.atan2(dy2, dx2))
  if abs(angle1-angle2)<8:
  # Kiểm tra xem các vector hướng có tỉ lệ thức bằng nhau không (coi như song song)
    if abs(dx1 * dy2 - dx2 * dy1) < tolerance:
      # Trường hợp đặc biệt: Kiểm tra xem các đường thẳng có cùng hoành độ vô cực hay không
      if dx1 == 0 and dx2 == 0:
        return True  # Cả hai đường thẳng đều thẳng đứng
      else:
        return True  # Các đường thẳng có cùng độ dốc
    elif abs(dx1 / dx2 - dy1 / dy2) < tolerance:
      return True
    else:
      return False
  else:
     return False
def calculate_parallelogram_area(A_point,B_point,C_point):
    # Tạo hai vectơ kề nhau
    vector1 = np.array([A_point[0] - B_point[0], A_point[1] - B_point[1]])
    vector2 = np.array([C_point[0] - B_point[0], C_point[1] - B_point[1]])
    # Tính tích có hướng
    area = np.abs(np.cross(vector1, vector2))

    return area
def intersection(line1, line2):
    x1,y1,x2,y2= line1
    x3,y3,x4,y4 = line2
    # Tính các hệ số của phương trình đường thẳng
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1*x1 + B1*y1
    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2*x3 + B2*y3

    # Giải hệ phương trình
    determinant = A1*B2 - A2*B1
    if determinant == 0:
        # Hai đường thẳng song song hoặc trùng nhau
        return None
    else:
        x = (B2*C1 - B1*C2)/determinant
        y = (A1*C2 - A2*C1)/determinant
        return x, y
    
def custom_sort(inner_list):
  """Sorts a sublist by the second element (ascending) and prioritizes larger first elements in case of ties."""
  return (inner_list[1], inner_list[0])  # Reverse first element for larger-first tie prioritization   
def find_bounding_box(matrix):
    rows, cols = np.where(matrix == 255)
    x_min, x_max = np.min(cols), np.max(cols)
    y_min, y_max = np.min(rows), np.max(rows)
    return x_min, y_min, x_max, y_max
def find_parallels_helper(lines, img, length, start_index, end_index):
    # time.sleep(1)
    parallel_lines=[]
    parallel_lines_ox = []
    angles_ox =[]
    parallel_lines_oy = []
    angles_oy =[]
    img_height,img_width,_ = img.shape
    print('qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq       ',start_index, end_index)
    # index=[]
    
# Do vẫn cần xét tất cả line nên chỉ chia luồng thành 4 phần nhỏ, 4 phần nhỏ sẽ so với cả lines chứ không phải cả line so với cả line như code đầu
    
    for i in range(start_index, end_index):
        for j in range(i+1,len(lines)):
            if is_parallel(lines[i][0],lines[j][0]):
                parallel_lines.append([lines[i][0],lines[j][0]])
                x1,y1,x2,y2 = lines[i][0]
                x3,y3,x4,y4 = lines[j][0]
                length1 = np.sqrt((x2 - x1)**2 + (y1 - y2)**2)
                length2 = np.sqrt((x3 - x4)**2 + (y3 - y4)**2)
                angle = np.degrees(math.atan2(y2-y1, x2-x1))
                if 45<angle<135 or -135<angle<-45:
                    dist = abs(min(x3,x4)-min(x1,x2))
                    if dist >= img_width/4 and length1 >= length and length2 >= length:
                        # index.append([i,j])
                        parallel_lines_oy.append([lines[i][0],lines[j][0]])
                        angles_oy.append(np.degrees(math.atan2(y2-y1, x2-x1)))
                else :
                    dist = abs(min(y3,y4)-min(y1,y2))
                    if dist >= img_height/4 and length1 >= length and length2 >= length:
                        parallel_lines_ox.append([lines[i][0],lines[j][0]])
                        angles_ox.append(np.degrees(math.atan2(y2-y1, x2-x1)))
    # print(index)
    print('ox',len(parallel_lines_ox),start_index)
    print('oy',len(parallel_lines_oy))
    return parallel_lines_ox,parallel_lines_oy
def find_parallels(lines, img, length):
    num_processes = 10  # Sử dụng số lượng CPU
    chunk_size = len(lines) // num_processes
    print(len(lines))
    
#Code chạy đa luồng với 4 luồng ở đây ạ
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(find_parallels_helper, [(lines, img, length, i*chunk_size, (i+1)*chunk_size) for i in range(num_processes)])

    # Kết hợp kết quả từ các tiến trình con
    all_parallel_lines_ox = []
    all_parallel_lines_oy = []
    for result_ox, result_oy in results:
        all_parallel_lines_ox.extend(result_ox)
        all_parallel_lines_oy.extend(result_oy)

    return all_parallel_lines_ox, all_parallel_lines_oy
def warped_images(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lsd = cv2.createLineSegmentDetector(3,2,2,3)

    lines= lsd.detect(gray)[0]

    img_height,img_width,_ = img.shape
    length = 15
    # results = find_parallels(lines,img,length)
    parallel_lines_ox,parallel_lines_oy = find_parallels(lines,img,length)
    # print('ox',len(parallel_lines_ox))
    # print('oy',len(parallel_lines_oy))
    # return parallel_lines_ox,parallel_lines_oy
    length_tmp=length
    while len(parallel_lines_ox) > 200:
        length_tmp+=5
        tmp=[]
        for line in parallel_lines_ox:
                x1,y1,x2,y2 = line[0]
                x3,y3,x4,y4 = line[1]
                length1 = np.sqrt((x2 - x1)**2 + (y1 - y2)**2)
                length2 = np.sqrt((x3 - x4)**2 + (y3 - y4)**2)
                if length1 >= length_tmp and length2 >= length_tmp:
                    tmp.append(line)
        parallel_lines_ox=None
        parallel_lines_ox=tmp
        print('ox',len(parallel_lines_ox))
        print('oy',len(parallel_lines_oy))
    length_tmp=length
    while len(parallel_lines_oy) > 200:
        length_tmp+=5
        tmp=[]
        for line in parallel_lines_oy:
                x1,y1,x2,y2 = line[0]
                x3,y3,x4,y4 = line[1]
                length1 = np.sqrt((x2 - x1)**2 + (y1 - y2)**2)
                length2 = np.sqrt((x3 - x4)**2 + (y3 - y4)**2)
                if length1 >= length_tmp and length2 >= length_tmp:
                    tmp.append(line)
        parallel_lines_oy=None
        parallel_lines_oy=tmp
        print('ox',len(parallel_lines_ox))
        print('oy',len(parallel_lines_oy))
    while len(parallel_lines_ox) == 0 or len(parallel_lines_oy) == 0:
        length-=10
        parallel_lines_ox,parallel_lines_oy = find_parallels(lines,img,length)

    max_area = 0
    max_parallelogram = None
    max_parallelogram_coordinate=None
    
    image_line = img.copy()
    for line in lines:
        x1,y1,x2,y2 = np.round(line[0]).astype(int)
        x1,y1,x2,y2 =int(x1),int(y1),int(x2),int(y2) 

        cv2.line(image_line,(x1,y1),(x2,y2),(0,0,255),2)


    count=0
    for i in range(len(parallel_lines_ox)):

        for j in range(len(parallel_lines_oy)):
                count+=1
            # if not is_parallel(parallel_lines[i][0],parallel_lines[j][0]):
                # print('i',i,'j',j)
                A_point = intersection(parallel_lines_ox[i][0],parallel_lines_oy[j][0])
                B_point = intersection(parallel_lines_oy[j][0],parallel_lines_ox[i][1])
                C_point = intersection(parallel_lines_ox[i][1],parallel_lines_oy[j][1])
                D_point = intersection(parallel_lines_oy[j][1],parallel_lines_ox[i][0])
                # print(A_point,B_point,C_point,D_point,"\n")
                if A_point and B_point and C_point and D_point:
                    points_width = [A_point[0],B_point[0],C_point[0],D_point[0]]
                    points_height = [A_point[1],B_point[1],C_point[1],D_point[1]]
                else:
                    continue
                if all(10<= num <= img_width-10 for num in points_width) and all(10 <= num <= img_height-10 for num in points_height):
                    x1,y1 = round(A_point[0]),round(A_point[1])
                    x2,y2 = round(B_point[0]),round(B_point[1])
                    x3,y3 = round(C_point[0]),round(C_point[1])
                    x4,y4 = round(D_point[0]),round(D_point[1])
                    area = calculate_parallelogram_area(A_point,B_point,C_point)
                    # print(area)
                    if area > max_area:
                        max_area = area
                        max_parallelogram = None
                        max_parallelogram = [parallel_lines_ox[i][0],parallel_lines_ox[i][1],parallel_lines_oy[j][0],parallel_lines_oy[j][1]]
                        # print(A_point,B_point,C_point,D_point,"\n")
                        
                        max_parallelogram_coordinate =[A_point,B_point,C_point,D_point]


    # print(max_parallelogram_coordinate)
    x1,y1 = round(max_parallelogram_coordinate[0][0]),round(max_parallelogram_coordinate[0][1])
    x2,y2 = round(max_parallelogram_coordinate[1][0]),round(max_parallelogram_coordinate[1][1])
    x3,y3 = round(max_parallelogram_coordinate[2][0]),round(max_parallelogram_coordinate[2][1])
    x4,y4 = round(max_parallelogram_coordinate[3][0]),round(max_parallelogram_coordinate[3][1])

    src_points = np.float32([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])


    sorted_values = sorted(src_points, key=custom_sort)
    if sorted_values[2][0] < sorted_values[3][0]:
        x1,y1 = sorted_values[2][0],sorted_values[2][1]
        x2,y2 = sorted_values[3][0],sorted_values[3][1]
    else:
        x1,y1 = sorted_values[3][0],sorted_values[3][1]
        x2,y2 = sorted_values[2][0],sorted_values[2][1]
    if sorted_values[0][0] < sorted_values[1][0]:
        x3,y3 = sorted_values[1][0],sorted_values[1][1]
        x4,y4 = sorted_values[0][0],sorted_values[0][1]
    else:
        x3,y3 = sorted_values[0][0],sorted_values[0][1]
        x4,y4 = sorted_values[1][0],sorted_values[1][1]
    # cv2.line(img,(round(x1),round(y1)),(round(x2),round(y2)),(255,0,0),2)
    # cv2.line(img,(round(x2),round(y2)),(round(x3),round(y3)),(255,0,255),2)
    # cv2.line(img,(round(x3),round(y3)),(round(x4),round(y4)),(0,255,0),2)
    # cv2.line(img,(round(x1),round(y1)),(round(x4),round(y4)),(0,0,255),2)
    test = np.float32([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
    x_min,x_max = min(x1,x2,x3,x4),max(x1,x2,x3,x4)
    y_min,y_max = max(y1,y2,y3,y4),min(y1,y2,y3,y4)
    dst_points = np.float32([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]])

    matrix = cv2.getPerspectiveTransform(test, dst_points)
    warped_image = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))
    print(img.shape[1], img.shape[0])
    for i in range(warped_image.shape[0]):
        for j in range(warped_image.shape[1]):
            if warped_image[i][j].shape[0]==3:
                comparison_array = np.array([200, 200, 200])
                if all(warped_image[i][j] <= comparison_array):
                    warped_image[i][j]=[255,255,255]
                else:
                    break
            else:
                comparison_array = np.array([200, 200, 200,200])
                if all(warped_image[i][j] <= comparison_array):
                    warped_image[i][j]=[255,255,255,255]
                else:
                    break
        for j in range(warped_image.shape[1]-1,-1,-1):
            if warped_image[i][j].shape[0]==3:
                comparison_array = np.array([200, 200, 200])
                if all(warped_image[i][j] <= comparison_array):
                    warped_image[i][j]=[255,255,255]
                else:
                    break
            else:
                comparison_array = np.array([200, 200, 200,200])
                if all(warped_image[i][j] <= comparison_array):
                    warped_image[i][j]=[255,255,255,255]
                else:
                    break
    return img,warped_image


# Tải dữ liệu
uploaded_file = st.file_uploader("Chọn file dữ liệu")
if uploaded_file is not None:
    img = np.array(Image.open(uploaded_file))
    # st.image(img, caption='Source image')
    time.sleep(2)
    # x1,x2 = warped_images(img)
    # print(len(x1))
    # print(len(x2))
    start_time = time.time()
    image,warped_image = warped_images(img)
    end_time = time.time()
    st.image(image, caption='Source image')
    st.image(warped_image, caption='Warped image')
    st.write("Thời gian chạy:", end_time - start_time, "giây")
    # img = Image.open(uploaded_file)
    # np_array = np.array(img)
    # img = cv2.imread(img)
    # image_path = uploaded_file.name
    # st.write(image_path)
    # result = run_pipeline(np_array)
    # result = read_image(np_array)
    # st.write(result)
    # print(result)
    # img_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # Tạo đối tượng Pillow Image
    # pil_image = Image.fromarray(img_rgb)

    # # Lưu ảnh dưới định dạng PIL
    # st.write(pil_image)
    # print(np_array)
    # st.write(uploaded_file)
