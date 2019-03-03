function [] = change()
load LandmineData.mat
whos
x = feature;
y = label;
[~, n] = size(x);
for i = 1:n
    x0 = x{1, i};
    y0 = y{1, i};
    name = sprintf('domain%d.txt', i);
    mat2txt(x0, y0, name);
end
end

