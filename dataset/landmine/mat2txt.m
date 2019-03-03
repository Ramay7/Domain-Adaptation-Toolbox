function back = mat2txt(x, y, name)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
fop = fopen(name, 'wt' );
[n, d] = size(x);
for i = 1:n
    for j = 1:d
        fprintf(fop, '%f ', x(i, j));
    end
    fprintf(fop, '%d\n', y(i, 1));
end
back = fclose(fop);
end

