import numpy as np 
import os 
import tensorflow as tf 
from matplotlib import pyplot as plt 
from PIL import Image 
import glob
import sys 
from object_detection.utils import visualization_utils as vis_util


# Helper code 
def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def main():

	MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous'

	# MODEL_NAME = 'faster_rcnn_resnet_101'
	# MODEL_NAME = 'faster_rcnn_resnet50'
	# MODEL_NAME = 'faster_rcnn_inception_v2'
	# MODEL_NAME = 'rfcn_resnet101'
	# MODEL_NAME = 'ssd_inception_v2'
	# MODEL_NAME = 'ssd_mobilenet_v1'

	# Path to frozen detection graph. This is the actual model that is used for the traffic sign detection.
	MODEL_PATH = os.path.join('models', MODEL_NAME)
	PATH_TO_CKPT = os.path.join(MODEL_PATH,'inference_graph/frozen_inference_graph.pb')

	# List of the strings that is used to add correct label for each box.
	# PATH_TO_LABELS = os.path.join('gtsdb_data', 'gtsdb3_label_map.pbtxt')
	PATH_TO_LABELS = 'gtsdb3_label_map.pbtxt'

	NUM_CLASSES = 3

	# Load a (frozen) Tensorflow model into memory
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.compat.v1.GraphDef()
		# od_graph_def = tf.GraphDef()
		# with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

	# Loading label map
	# with open("scripts/data.json", "r") as read_file:
	# 	# data = json.load(read_file)
	# 	data = read_file.read()
	
	# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)


	# categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)

	categories = []
	
	categories.append({ 'id': 1, 'name': 'prohibitory'})
	categories.append({ 'id': 2, 'name': 'mandatory'})
	categories.append({ 'id': 3, 'name': 'danger'})
	
	category_index = {}
	for cat in categories:
		category_index[cat['id']] = cat

	# for class_id in range(NUM_CLASSES):
	# 	categories.append({
	# 		'id': class_id + label_id_offset,
	# 		'name': 'category_{}'.format(class_id + label_id_offset)
	# 	})

	# category_index = label_map_util.create_category_index(categories)
	# print(label_map)


	# Detection Code
	# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
	PATH_TO_TEST_IMAGES = '../data/test1'
	TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES, '*.jpg'))

	# Size, in inches, of the output images.
	IMAGE_SIZE = (20, 20)


	with detection_graph.as_default():
		with tf.compat.v1.Session(graph=detection_graph) as sess:
			for idx, image_path in enumerate(TEST_IMAGE_PATHS):
				image = Image.open(image_path)
				print("The image size: ", image.size)
				print("\n")
				image = image.resize((1360, 800))
				# 
				# the array based representation of the image will be used later in order to prepare the
				# result image with boxes and labels on it.
				image_np = load_image_into_numpy_array(image)
				# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
				image_np_expanded = np.expand_dims(image_np, axis=0)
				image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
				# Each box represents a part of the image where a particular object was detected.
				boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
				# Each score represent how level of confidence for each of the objects.
				# Score is shown on the result image, together with the class label.
				scores = detection_graph.get_tensor_by_name('detection_scores:0')
				classes = detection_graph.get_tensor_by_name('detection_classes:0')
				num_detections = detection_graph.get_tensor_by_name('num_detections:0')
				# Actual detection.
				(boxes, scores, classes, num_detections) = sess.run(
					[boxes, scores, classes, num_detections],
					feed_dict={image_tensor: image_np_expanded})
				# Visualization of the results of a detection.
				vis_util.visualize_boxes_and_labels_on_image_array(
					image_np,
					np.squeeze(boxes),
					np.squeeze(classes).astype(np.int32),
					np.squeeze(scores),
					category_index,
					use_normalized_coordinates=True,
					line_thickness=6)
				plt.figure(idx, figsize=IMAGE_SIZE)
				plt.axis('off')
				plt.imshow(image_np)
				plt.savefig("outputs/out"+str(idx)+".png")


if __name__ == "__main__":
	main()
