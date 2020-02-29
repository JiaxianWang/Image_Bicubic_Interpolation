
I = imread('C:\Users\BRUCE WANG\PycharmProjects\Bicubic_Inter\img_example_lr.png');

I = double(I);

[oh, ow, od] = size(I);

zmf = 2; 

% initialtargetimageTI

th = round(oh * zmf);

tw = round(ow * zmf);

TI = zeros(th, tw, od); 

% addoriginalimagewith 2 rows and 2 cols


a = I(1,:,:); b = I(oh,:,:);

temp_I = [a;
a;
I;
b;
b];

c = temp_I(:, 1,:); d = temp_I(:, ow,:);

FI = [c, c, temp_I, d, d];


for w = 1:tw

j = floor(w / zmf) + 2;
v = rem(w, zmf) / zmf;

for h = 1:th

i = floor(h / zmf) + 2;
u = rem(h, zmf) / zmf;

A = [s(u + 1), s(u), s(u - 1), s(u - 2)];

C = [s(v + 1);
s(v);
s(v - 1);
s(v - 2)];

for d = 1:od % image's 3 channels

B = FI(i - 1:i + 2, j - 1: j + 2, d);

TI(h, w, d) = A * B * C;

end

end

end

figure;

imshow(uint8(TI));

toc;



function w = s(wx)

    wx = abs(wx);

    if wx<1

        w = 1 - 2*wx^2 + wx^3;

    elseif wx>=1 && wx<2

        w = 4 - 8*wx + 5*wx^2 - wx^3;

    else

        w = 0;

    end
end