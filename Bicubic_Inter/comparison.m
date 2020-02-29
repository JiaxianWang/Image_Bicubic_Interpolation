
imdata_g = imread('C:\Users\BRUCE WANG\PycharmProjects\Bicubic_Inter\image_example_matlab.png');


ref_g=rgb2gray(imdata_g);

imdata_i = imread('C:\Users\BRUCE WANG\PycharmProjects\Bicubic_Inter\img_example_lr_test.png');


ref_i=rgb2gray(imdata_i);
error = immse(ref_g, ref_i)