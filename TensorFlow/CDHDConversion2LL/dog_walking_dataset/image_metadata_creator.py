from PIL import Image
import glob

def main():
	prototypical_lst = glob.glob("DogWalkingLittleLandmarks/prototypical/*.labl")
	
	for image in prototypical_lst:
		try:
			crop_prototypical_image(image[39:])
		except ValueError as e:
			#Exception will occur if 'attached-to-leash' and 'holding-leash" is not in label file
			print(image[39:])
			print str(e)


def crop_prototypical_image(labl_filename):
	file_object  = open("DogWalkingLittleLandmarks/prototypical/" + labl_filename, "r") 
	labl = file_object.read()

	labl_lst = labl.split("|")

	labl_count = int(labl_lst[2])

	j = 3
	coord_tupl_lst = []
	labl_tupl_lst = []
	for i in range (0,labl_count):
		coord_tupl_lst.append((labl_lst[j],labl_lst[j+1],labl_lst[j+2],labl_lst[j+3]))
		j = j + 4

	for i in range (0,labl_count):
		if(labl_lst[j].startswith('leash')):
			labl_tupl_lst.append('leash')
		else:
			labl_tupl_lst.append(labl_lst[j])
		j = j + 1

	holding_leash_idx = labl_tupl_lst.index('holding-leash')
	ll1_bb_coords = tuple(map(int, coord_tupl_lst[holding_leash_idx]))
	ll1_cent_coords = (ll1_bb_coords[0] + (ll1_bb_coords[2]/2), ll1_bb_coords[1] + (ll1_bb_coords[3]/2))

	attached_to_leash_idx = labl_tupl_lst.index('attached-to-leash')
	ll2_bb_coords = tuple(map(int, coord_tupl_lst[attached_to_leash_idx]))
	ll2_cent_coords = (ll2_bb_coords[0] + (ll2_bb_coords[2]/2), ll2_bb_coords[1] + (ll2_bb_coords[3]/2))

	leash_bb_idx = labl_tupl_lst.index('leash')
	leash_bb_coords = tuple(map(int, coord_tupl_lst[leash_bb_idx]))
	leash_cent_coords = (leash_bb_coords[0], leash_bb_coords[1], leash_bb_coords[0] + leash_bb_coords[2], leash_bb_coords[1] + leash_bb_coords[3])

	image_name = "DogWalkingLittleLandmarks/prototypical/" + labl_filename[:labl_filename.index(".")] + ".jpg"

	out_str = str(ll1_cent_coords[0]) + '|' + str(ll1_cent_coords[1]) + '|' + \
			  str(ll2_cent_coords[0]) + '|' + str(ll2_cent_coords[1]) + '|' + \
			  image_name + '|' + \
			  str(labl_lst[0]) + ',' + str(labl_lst[1]) + ',' + str(3) + '|' + \
			  str(leash_cent_coords[0]) + ',' + str(leash_cent_coords[1]) + ',' + str(leash_cent_coords[2]) + ',' + str(leash_cent_coords[3])

	out_f = open('dwi_anno_data.txt', 'a+')	
	out_f.write(out_str + '\n')		  
	out_f.close()

if __name__ == "__main__":
    main()