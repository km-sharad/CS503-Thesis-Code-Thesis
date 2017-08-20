file_name='/home/sharad/CS503-Thesis/car_dataset/cdhd_anno.mat'
anno_cdhd = load(file_name)
anno_cdhd.cdhd(1,1).coords(1)
fileID = fopen('cdhd_anno.txt','w');
for i = 1 :length(anno_cdhd.cdhd)
    str = strcat(num2str(anno_cdhd.cdhd(1,i).coords(1)),'|',num2str(anno_cdhd.cdhd(1,i).coords(2)),'|',anno_cdhd.cdhd(1,i).filepath,'|',num2str(anno_cdhd.cdhd(1,i).imgdims(1)),',',num2str(anno_cdhd.cdhd(1,i).imgdims(2)),',',num2str(anno_cdhd.cdhd(1,i).imgdims(3)),'|',num2str(anno_cdhd.cdhd(1,i).istrain),'|',num2str(anno_cdhd.cdhd(1,i).istest),'|',num2str(anno_cdhd.cdhd(1,i).is_visible),'|',num2str(anno_cdhd.cdhd(1,i).class),'|',num2str(anno_cdhd.cdhd(1,i).bbox(1)),',',num2str(anno_cdhd.cdhd(1,i).bbox(2)),',',num2str(anno_cdhd.cdhd(1,i).bbox(3)),',',num2str(anno_cdhd.cdhd(1,i).bbox(4)))
    fprintf(fileID,'%s\n',str);
end
