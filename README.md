# ahead

we shuld have some folders

--util
	universe tools include image process

--config
	all params

--data_files
	weights file and data file

--models
	algo model

--demo
	some use method of this arch
	and adding request script

--pipeline
	the whole process


pipeline
	
	1 read the big image and cut it with 1216*1216

		in: 
			slide obj: 		one kfb or tiff image
		out:
			list: 			dict 1216
							x 		: top left corner coordinate x
							y 		: top left corner coordinate y
							w 		: width of the image
							h  		: height of the image
							img_data 	: numpy image data 1216
							img_pre 	: numpy image data 299

	2 run the first predict algo

		in:
			1's dict list
		out:
			1) bool:		is_normal

			2) list: 		
				dict 1216
					x 		: top left corner coordinate x
					y 		: top left corner coordinate y
					w 		: width of the image
					h  		: height of the image							
					img_data 	: numpy image data 1216
					img_pre 	: numpy image data 299
					is_normal	: true or false

			3) int: 		normal nums

			4) int: 		abnormal nums

	if 2 retuen normal, the program should return normal, pipeline end

	3 cut 2's result, from 1216*1216 to 608*608

		in:
			2's dict list
		out:
			list: 		
				dict 608
					x 		: top left corner coordinate x
					y 		: top left corner coordinate y
					w 		: width of the image
					h  		: height of the image
					img_data 	: numpy image data 608
					index       : 608 images index
			

	4 run the yolo cell detect algo
		in:
			2's dict list
		out:
			list:
				dict cell
					x 			: top left corner coordinate x
					y 			: top left corner coordinate y
					w 			: width of the image
					h  			: height of the image
					yolo_class 	:
					det 			:
					index 		: from which 608 image

	5 cut 4's result, from 608 dict list to cell

		in:
			1) 4's cell dict list
			2) 3's 608 image dict list

		out:
			list:
				dict cell
					x 			: top left corner coordinate x
					y 			: top left corner coordinate y
					w 			: width of the image
					h  			: height of the image
					yolo_class 	:
					det 			:
					index 		: from which 608 image
					cell_data 		: numpy image data 299

	6 run the cell classify algo

		in:
			5's dict list

		out:
			list:
				dict cell
					x 			: top left corner coordinate x
					y 			: top left corner coordinate y
					w 			: width of the image
					h  			: height of the image
					yolo_class 	:
					det 			:
					index 		: from which 608 image
					cell_data 		: numpy image data 299
					xcp_class 		:

	7 run the diagnosis algo

		in:
			6's dict list

		out:
			list:
				dict
					probability 	:
					label class 	:










