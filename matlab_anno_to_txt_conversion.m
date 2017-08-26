%coords: 2D coordinates of the front door handle
%filepath: Path of the image relative to a root containing the images dir.
%imgdims: Size of the image
%istrain: true if it is a training image (uses the train/test split of original dataset).
%istest: true if it is a testing image
%is_visible: true if door handle is visible. Note that for some of the images with a frontal view of the car, annotators marked them as invisible. Such images were ignored during training/testing.
%class: Integral id of the class the image belongs to.    
%bbox_x1: Min x-value of the bounding box, in pixels
%bbox_x2: Max x-value of the bounding box, in pixels
%bbox_y1: Min y-value of the bounding box, in pixels
%bbox_y2: Max y-value of the bounding box, in pixels

file_name='/home/sharad/CS503-Thesis/car_dataset/cdhd_anno.mat'
anno_cdhd = load(file_name)
anno_cdhd.cdhd(1,1).coords(1)
fileID = fopen('cdhd_anno.txt','w');
for i = 1 :length(anno_cdhd.cdhd)
    str = strcat(num2str(anno_cdhd.cdhd(1,i).coords(1)),'|',num2str(anno_cdhd.cdhd(1,i).coords(2)),'|',anno_cdhd.cdhd(1,i).filepath,'|',num2str(anno_cdhd.cdhd(1,i).imgdims(2)),',',num2str(anno_cdhd.cdhd(1,i).imgdims(1)),',',num2str(anno_cdhd.cdhd(1,i).imgdims(3)),'|',num2str(anno_cdhd.cdhd(1,i).istrain),'|',num2str(anno_cdhd.cdhd(1,i).istest),'|',num2str(anno_cdhd.cdhd(1,i).is_visible),'|',num2str(anno_cdhd.cdhd(1,i).class),'|',num2str(anno_cdhd.cdhd(1,i).bbox(1)),',',num2str(anno_cdhd.cdhd(1,i).bbox(2)),',',num2str(anno_cdhd.cdhd(1,i).bbox(3)),',',num2str(anno_cdhd.cdhd(1,i).bbox(4)))
    fprintf(fileID,'%s\n',str);
end
