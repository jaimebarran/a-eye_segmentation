'''
read .annot file from ITK-SNAP
'''

__author__ = "Jaime Barranco"
__email__ = "jaimebarran@gmail.com"
# __version__ = "0.0.1"
# __status__ = "Prototype"


import xml.etree.ElementTree as ET
import numpy as np


def create_dict_from_xml(path):

    tree = ET.parse(path)
    root = tree.getroot()

    list_values = list()
    # print all tags and attributes
    for elem in root.iter():
        # print(elem.tag, elem.attrib)
        # create list with all the attributes
        for key, value in elem.attrib.items():
            list_values.append(value)

    # create dictionary
    # TODO: for several dictionaries (one per annotation),
    # check this key: <entry key="ArraySize" value="2" />
    dict_aux = dict()
    for i in range(0, len(list_values), 2):
        dict_aux[list_values[i]] = list_values[i+1]
    
    return dict_aux

def extract_points(dictionary):
    if dictionary['Type'] == 'LineSegmentAnnotation' and dictionary['Plane'] == '2':
        point1 = np.fromstring(dictionary['Point1'], dtype=float, sep=' ')
        point2 = np.fromstring(dictionary['Point2'], dtype=float, sep=' ')
        return point1, point2

def distance(point1, point2):
    return np.linalg.norm(point1 - point2)


xml_path = '/home/jaimebarranco/Desktop/0001.annot'
# xml_path = '/home/jaimebarranco/Downloads/measure.annot'

# create dictionary
dict_annot = create_dict_from_xml(xml_path)

# extract points
point1, point2 = extract_points(dict_annot)

# compute distance
axial_length = np.around(distance(point1, point2), decimals=2)
print(f'Axial lenght = {axial_length} mm')
