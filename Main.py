import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import os
import math
import multiprocessing
import concurrent.futures
# from numba import njit,prange
# from numba.np.extensions import cross2d
# from image_enhancement.run import *
# @njit(fastmath=True, cache=True)
def is_parallel(line1, line2, tolerance=3):
  dx1 = line1[2] - line1[0]
  dy1 = line1[3] - line1[1]
  dx2 = line2[2] - line2[0]
  dy2 = line2[3] - line2[1]
  angle1=np.degrees(math.atan2(dy1, dx1))
  angle2=np.degrees(math.atan2(dy2, dx2))
  if abs(angle1-angle2)<8:
    if abs(dx1 * dy2 - dx2 * dy1) < tolerance:
      if dx1 == 0 and dx2 == 0:
        return True 
      else:
        return False
    elif dx2!=0 and dy2!=0 and abs(dx1 / dx2 - dy1 / dy2) < tolerance:
      return True
    else:
      return False
  else:
     return False
# @njit(fastmath=True, cache=True)
def calculate_parallelogram_area(A_point,B_point,C_point):

    vector1 = np.array([A_point[0] - B_point[0], A_point[1] - B_point[1]])
    vector2 = np.array([C_point[0] - B_point[0], C_point[1] - B_point[1]])

    area = np.abs(vector1[0] * vector2[1] - vector1[1] * vector2[0])

    return area
# @njit(fastmath=True, cache=True)
def intersection(line1, line2):
    x1,y1,x2,y2= line1
    x3,y3,x4,y4 = line2

    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1*x1 + B1*y1
    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2*x3 + B2*y3

    determinant = A1*B2 - A2*B1
    if determinant == 0:
        return None
    else:
        x = (B2*C1 - B1*C2)/determinant
        y = (A1*C2 - A2*C1)/determinant
        return x, y
# @njit(fastmath=True, cache=True)   
def custom_sort(inner_list):
  return (inner_list[1], inner_list[0])  
# @njit(fastmath=True, cache=True, parallel=True)
def angle_line(line):
    x1, y1, x2, y2 = line[0]
    angle = np.degrees(math.atan2(y2 - y1, x2 - x1))
    return angle
def find_parallels_helper(lines, img, length, start_index, end_index):
    max_diff_angle=8
    parallel_lines_ox = []
    parallel_lines_oy = []
    img_height, img_width, _ = img.shape
    angle_all = [angle_line(lines[i]) for i in range(len(lines))]
    for i in range(start_index, end_index):  # Use prange for parallelism
        lines_potiental = [line for line in lines if abs(angle_line(lines[i]) - angle_line(line)) <= max_diff_angle]

        # print('line potential',len(lines_potiental))
        for j in range(len(lines_potiental)):
            if is_parallel(lines[i][0], lines_potiental[j][0]):
                x1, y1, x2, y2 = lines[i][0]
                x3, y3, x4, y4 = lines_potiental[j][0]
                length1 = np.sqrt((x2 - x1)**2 + (y1 - y2)**2)
                length2 = np.sqrt((x3 - x4)**2 + (y3 - y4)**2)
                angle = np.degrees(math.atan2(y2 - y1, x2 - x1))
                if 45 < angle < 135 or -135 < angle < -45:
                    dist = abs(min(x3, x4) - min(x1, x2))
                    if dist >= img_width / 4 and length1 >= length and length2 >= length:
                        parallel_lines_oy.append([lines[i][0], lines_potiental[j][0]])
                else:
                    dist = abs(min(y3, y4) - min(y1, y2))
                    if dist >= img_height / 4 and length1 >= length and length2 >= length:
                        parallel_lines_ox.append([lines[i][0], lines_potiental[j][0]])
    # # print('qqqqqqqqqqqqqq  ox',len(parallel_lines_ox),start_index,end_index)
    # # print('qqqqqqqqqqqqqq  oy',len(parallel_lines_oy))
    return parallel_lines_ox, parallel_lines_oy
def find_parallels(lines, img, length):
    num_processes = 10  # Sử dụng số lượng CPU
    chunk_size = len(lines) // num_processes
    # print('LLLLLLLLLEEEEEEEEEEEENNNNNNNN',len(lines))
    
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
def filter_lines_by_length(lines, length_tmp):
    tmp = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x3, y3, x4, y4 = line[1]
        length1 = np.sqrt((x2 - x1)**2 + (y1 - y2)**2)
        length2 = np.sqrt((x3 - x4)**2 + (y3 - y4)**2)
        if length1 >= length_tmp and length2 >= length_tmp:
            tmp.append(line)
    return tmp

def update_parallel_lines(parallel_lines, initial_length):
    length_tmp = initial_length
    while len(parallel_lines) > 150:
        length_tmp += 10
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Chia dữ liệu thành các phần nhỏ hơn để xử lý đa luồng
            num_splits = 4
            split_size = len(parallel_lines) // num_splits
            futures = []
            results = []
            for i in range(num_splits):
                start_index = i * split_size
                end_index = None if i == num_splits - 1 else (i + 1) * split_size
                chunk = parallel_lines[start_index:end_index]
                futures.append(executor.submit(filter_lines_by_length, chunk, length_tmp))

            # Kết hợp kết quả từ tất cả các luồng
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        parallel_lines = results
    return parallel_lines
def warped_images(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lsd = cv2.createLineSegmentDetector(3,2,2,3)

    lines= lsd.detect(gray)[0]
    line_lengths=[]
    for line in lines:
        x1,y1,x2,y2 = line[0]
        line_length = np.sqrt((x2 - x1)**2 + (y1 - y2)**2)
        line_lengths.append(line_length)
    combined = list(zip(line_lengths, lines))

    # Sắp xếp danh sách combined theo y giảm dần
    combined_sorted = sorted(combined, key=lambda pair: pair[0], reverse=True)

    # Tách phần x đã sắp xếp từ combined_sorted
    lines_sorted = [pair[1] for pair in combined_sorted]
    line_tests=[]
    for i in range(int(len(lines_sorted)*0.3)):
        line_tests.append(lines_sorted[i])
    # # print(line_lengths)
    # # print(lines_sorted)
    img_height,img_width,_ = img.shape
    length = 25
    # results = find_parallels(lines,img,length)
    parallel_lines_ox,parallel_lines_oy = find_parallels(line_tests,img,length)
    print('ox',len(parallel_lines_ox))
    print('oy',len(parallel_lines_oy))
    # return parallel_lines_ox,parallel_lines_oy
    if len(parallel_lines_ox) == 0 or len(parallel_lines_oy) == 0:
        length=0
        parallel_lines_ox,parallel_lines_oy = find_parallels(lines,img,length)
    print('LENGTHHHHHHHHHHHHHHHH',length)
    # parallel_lines_ox=shorten_lines(parallel_lines_ox,length_tmp)
    # for line in parallel_lines_ox:
    #     x1,y1,x2,y2 = np.round(line[0]).astype(int)
    #     x1,y1,x2,y2 =int(x1),int(y1),int(x2),int(y2) 

    #     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    length_tmp=length
    # while len(parallel_lines_ox) > 150:
    #     length_tmp+=10
    #     tmp=[]
    #     for line in parallel_lines_ox:
    #             x1,y1,x2,y2 = line[0]
    #             x3,y3,x4,y4 = line[1]
    #             length1 = np.sqrt((x2 - x1)**2 + (y1 - y2)**2)
    #             length2 = np.sqrt((x3 - x4)**2 + (y3 - y4)**2)
    #             if length1 >= length_tmp and length2 >= length_tmp:
    #                 tmp.append(line)
    #     parallel_lines_ox=None
    #     parallel_lines_ox=tmp
    #     # print('ox',len(parallel_lines_ox))
    #     # print('oy',len(parallel_lines_oy))
    # length_tmp=length
    # while len(parallel_lines_oy) > 150:
    #     length_tmp+=10
    #     tmp=[]
    #     for line in parallel_lines_oy:
    #             x1,y1,x2,y2 = line[0]
    #             x3,y3,x4,y4 = line[1]
    #             length1 = np.sqrt((x2 - x1)**2 + (y1 - y2)**2)
    #             length2 = np.sqrt((x3 - x4)**2 + (y3 - y4)**2)
    #             if length1 >= length_tmp and length2 >= length_tmp:
    #                 tmp.append(line)
    #     parallel_lines_oy=None
    #     parallel_lines_oy=tmp
        # print('ox',len(parallel_lines_ox))
        # print('oy',len(parallel_lines_oy))
    start_time = time.time()
    parallel_lines_ox = update_parallel_lines(parallel_lines_ox, length_tmp)
    parallel_lines_oy = update_parallel_lines(parallel_lines_oy, length_tmp)
    end_time = time.time()
    print(end_time-start_time,'s')
    max_area = 0
    max_parallelogram_coordinate=None
    
    image_line = img.copy()
    # for line in parallel_lines_ox:
    #     x1,y1,x2,y2 = np.round(line[0]).astype(int)
    #     x1,y1,x2,y2 =int(x1),int(y1),int(x2),int(y2) 

    #     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)


    count=0
    for i in range(len(parallel_lines_ox)):

        for j in range(len(parallel_lines_oy)):
                count+=1
            # if not is_parallel(parallel_lines[i][0],parallel_lines[j][0]):
                # # print('i',i,'j',j)
                A_point = intersection(parallel_lines_ox[i][0],parallel_lines_oy[j][0])
                B_point = intersection(parallel_lines_oy[j][0],parallel_lines_ox[i][1])
                C_point = intersection(parallel_lines_ox[i][1],parallel_lines_oy[j][1])
                D_point = intersection(parallel_lines_oy[j][1],parallel_lines_ox[i][0])
                # # print(A_point,B_point,C_point,D_point,"\n")
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
                    # # print(area)
                    if area > max_area:
                        max_area = area
                        max_parallelogram = None
                        max_parallelogram = [parallel_lines_ox[i][0],parallel_lines_ox[i][1],parallel_lines_oy[j][0],parallel_lines_oy[j][1]]
                        # # print(A_point,B_point,C_point,D_point,"\n")
                        
                        max_parallelogram_coordinate =[A_point,B_point,C_point,D_point]


    # # print(max_parallelogram_coordinate)
    x1,y1 = round(max_parallelogram_coordinate[0][0]),round(max_parallelogram_coordinate[0][1])
    x2,y2 = round(max_parallelogram_coordinate[1][0]),round(max_parallelogram_coordinate[1][1])
    x3,y3 = round(max_parallelogram_coordinate[2][0]),round(max_parallelogram_coordinate[2][1])
    x4,y4 = round(max_parallelogram_coordinate[3][0]),round(max_parallelogram_coordinate[3][1])

    src_points = np.float32([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])


    sorted_values = sorted(src_points, key=custom_sort)
    # # print(src_points)
    # # print(sorted_values)
    # x1,y1 = sorted_values[0][0],sorted_values[0][1]
    # x2,y2 = sorted_values[1][0],sorted_values[1][1]
    # x3,y3 = sorted_values[2][0],sorted_values[2][1]
    # x4,y4 = sorted_values[3][0],sorted_values[3][1]
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
    dist_width = np.sqrt((x1 - x2 )**2 + (y1  - y2 )**2)
    dist_height = np.sqrt((x1  - x4 )**2 + (y1 - y4 )**2)
    # if dist_width < dist_height:
    #     x_tmp,y_tmp = x4,y4
    #     x4,y4 = x3,y3
    #     x3,y3 = x2,y2
    #     x2,y2 = x1,y1
    #     x1,y1 = x_tmp,y_tmp
    #     dist_height,dist_width = dist_width,dist_height
    # cv2.line(img,(round(x1),round(y1)),(round(x2),round(y2)),(255,0,0),2)
    # cv2.line(img,(round(x2),round(y2)),(round(x3),round(y3)),(255,0,255),2)
    # cv2.line(img,(round(x3),round(y3)),(round(x4),round(y4)),(0,255,0),2)
    # cv2.line(img,(round(x1),round(y1)),(round(x4),round(y4)),(0,0,255),2)
    # cv2.putText(img, 'A', (round(x1),round(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # cv2.putText(img, 'B', (round(x2),round(y2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # cv2.putText(img, 'C', (round(x3),round(y3)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # cv2.putText(img, 'D', (round(x4),round(y4)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    test = np.float32([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
    x_min,x_max = min(x1,x2,x3,x4),max(x1,x2,x3,x4)
    y_min,y_max = max(y1,y2,y3,y4),min(y1,y2,y3,y4)
    # print(x1,y1)
    # dst_points = np.float32([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]])
    dst_points = np.float32([[x_min,y_min],[x_min+dist_width,y_min],[x_min+dist_width,y_min-dist_height],[x_min,y_min-dist_height]])

    matrix = cv2.getPerspectiveTransform(test, dst_points)
    warped_image = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))
    # # print(img.shape[1], img.shape[0])
    for i in range(warped_image.shape[0]):
        for j in range(warped_image.shape[1]):
            # # print('shape ',warped_image[i][j].shape[0])
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
    # # print(len(x1))
    # # print(len(x2))
    start_time = time.time()
    image,warped_image = warped_images(img)
    end_time = time.time()
    st.image(image, caption='Source image')
    st.image(warped_image, caption='Warped image')
    st.markdown(f'<p style="display: inline;font-size:24px;">Thời gian chạy: <div style="display: inline;color:green;font-size:30px;"> {end_time - start_time}</div> giây</p>', unsafe_allow_html=True)
    # img = Image.open(uploaded_file)
    # np_array = np.array(img)
    # img = cv2.imread(img)
    # image_path = uploaded_file.name
    # st.write(image_path)
    # result = run_pipeline(np_array)
    # result = read_image(np_array)
    # st.write(result)
    # # print(result)
    # img_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # Tạo đối tượng Pillow Image
    # pil_image = Image.fromarray(img_rgb)

    # # Lưu ảnh dưới định dạng PIL
    # st.write(pil_image)
    # # print(np_array)
    # st.write(uploaded_file)
