# Script to fit the polynomial depth
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys
import sympy as sp
from tqdm import tqdm
import json

if len(sys.argv) == 1:
    print("Please specify some data to analyze (python3 analyze_data.py simulation1.csv simulation2.csv)")
    quit()

data = []
for fname in sys.argv[1:]:
    with open(fname, "r") as f:
        data += [list(map(float, line.split(","))) for line in f.readlines()]

min_x = min_y = 0  # given the screen size, these are the max/min possible pixel positions of the bounding box edges
max_x = 599  # cropped phone screen resolution
max_y = 799
delta = 8   # for determining ratios, how far from the edge can the point be

IN_CENTER = 0
LEFT_EDGE = 1
TOP_EDGE = 2
RIGHT_EDGE = 4
BOTTOM_EDGE = 8

UP_RIGHT_CORNER = RIGHT_EDGE + TOP_EDGE
UP_LEFT_CORNER = LEFT_EDGE + TOP_EDGE
DOWN_RIGHT_CORNER = RIGHT_EDGE + BOTTOM_EDGE
DOWN_LEFT_CORNER = LEFT_EDGE + BOTTOM_EDGE

eps = 1e-4
def which_edges(point):
    edge = 0
    if abs(point[0] - min_x) < eps:
        edge += LEFT_EDGE
    if abs(point[1] - min_y) < eps:
        edge += BOTTOM_EDGE
    if abs(point[2] - max_x) < eps:
        edge += RIGHT_EDGE
    if abs(point[3] - max_y) < eps:
        edge += TOP_EDGE
    return edge

#print(len(data))
clean_data = list(filter(lambda el: el[2] >= 0, data))
#print(len(clean_data))
#quit()
in_center_data = np.array(list(filter(lambda el: which_edges(el) == IN_CENTER, clean_data))).T

bounding_boxes = np.zeros((2, in_center_data.shape[1]))
bounding_boxes[0] = in_center_data[2] - in_center_data[0]
bounding_boxes[1] = in_center_data[3] - in_center_data[1]
good_spots = np.nonzero(np.product(bounding_boxes,axis=0))[0]

def close_to_edge(pt):
    return ((pt[0] < delta) + (pt[1] < delta) + (pt[2] > max_x-delta) + (pt[3] > max_y-delta)) > 0
good_edge_points = good_spots[np.where(close_to_edge(in_center_data[:,good_spots]))]
print(good_edge_points, good_edge_points.shape, good_spots.shape)
ratios = bounding_boxes[1][good_edge_points]/bounding_boxes[0][good_edge_points]  # height:width
ratio_features = np.zeros((6, ratios.shape[0]))
ratio_features[0:4] = in_center_data[0:4,good_edge_points]
ratio_features[4:6] = bounding_boxes[:,good_edge_points]   # used for min distance clustering later

max_ratio = ratios.max()
min_ratio = ratios.min()
print("Min, max ratios:", min_ratio, max_ratio)
#plt.hist(ratios)
#plt.show()

def _extended_bounding_box(point):
    width = point[2]-point[0]
    height = point[3]-point[1]
    bbox = [width, height]
    edge = which_edges(point)

    if edge == LEFT_EDGE:
        point_feature = np.array([point[2], point[3], height])  # these are the coordinates that we know for sure
        feature_indices = [2,3,5]
        bbox_item = 0 # want to change the width
    elif edge == RIGHT_EDGE:
        point_feature = np.array([point[0], point[1], height])  # these are the coordinates that we know for sure
        feature_indices = [0,1,5]
        bbox_item = 0
    elif edge == TOP_EDGE:
        point_feature = np.array([point[0], point[1], width])  # these are the coordinates that we know for sure
        feature_indices = [0,1,4]
        bbox_item = 1
    elif edge == BOTTOM_EDGE:
        point_feature = np.array([point[2], point[3], width])  # these are the coordinates that we know for sure
        feature_indices = [2,3,4]
        bbox_item = 1
    elif edge in [UP_LEFT_CORNER, UP_RIGHT_CORNER, DOWN_LEFT_CORNER, DOWN_RIGHT_CORNER]:
        #print("CORNER >:( >:( >:( >:( >:( >:( >:( >:( >:( >:( >:( >:( >:( >:( >:( >:( >:( >:( >:( >:( >:( ")
        return None     #corner cases, ignore for now
    if edge != IN_CENTER:   # always compute this for points not in the center
        closest = np.argmin(((np.expand_dims(point_feature,1) - ratio_features[feature_indices])**2).sum(axis=0))
        desired_ratio = ratios[closest]
        bbox[bbox_item] = bbox[1-bbox_item]/desired_ratio
    if bbox[0] == 0 or bbox[1] == 0:
        return None
    return bbox

full_data = []
for point in tqdm(clean_data):
    bbox = _extended_bounding_box(point[:-1])
    if bbox is not None:
        full_data.append([*bbox, point[4]])
full_data = np.array(full_data)
print(f"Found {full_data.shape[0]} good data points")
x_data = full_data[:, :2]
y_data = full_data[:, -1]

# these are useful regardless of the order of the model
def build_lin_res(data):
    x_lin_res = np.ones((3, data.shape[0]))
    x_lin_res[1] = data[:, 0]
    x_lin_res[2] = data[:, 1]
    return x_lin_res
x_lin_residue = build_lin_res(x_data)

def build_quad_res(data):
    x_quad_res = np.ones((4, data.shape[0]))
    x_quad_res[1] = data[:, 0]**2
    x_quad_res[2] = data[:, 1]**2
    x_quad_res[3] = data[:, 1]*data[:, 0]
    return x_quad_res
x_quad_residue = build_quad_res(x_data)

parse_str = lambda s: sp.parse_expr(s, transformations=(sp.parsing.sympy_parser.standard_transformations + (sp.parsing.sympy_parser.implicit_multiplication_application,)))

def add_term(w_exp, h_exp, features, _expr, _i, data):
    features[_i] = data[:,0]**w_exp * data[:,1]**h_exp
    _i += 1
    _expr += " + " + str(parse_str(f"(w**{w_exp})*(h**{h_exp})"))
    return _expr, _i

def build_features(order, neg_only, data):
    num_poly = 1 + 2*(order)*(order+1)
    if neg_only: # note that num_poly only includes non-residue terms
        num_poly -= ((order+1)*(order+2)//2-1)

    x_featured = np.ones((num_poly, data.shape[0]))
    i = 1
    expr = "1"

    for num in range(1,order+1):
        for shift in range(num+1):
            w_exponent = shift
            h_exponent = num-shift
            if not neg_only:
                expr,i = add_term(w_exponent, h_exponent, x_featured, expr, i, data)
            if w_exponent != 0:
                expr,i = add_term(-w_exponent, h_exponent, x_featured, expr, i, data)
            if h_exponent != 0:
                expr,i = add_term(w_exponent, -h_exponent, x_featured, expr, i, data)
            if w_exponent != 0 and h_exponent != 0:
                expr,i = add_term(-w_exponent, -h_exponent, x_featured, expr, i, data)
    return x_featured, expr, num_poly

def mse_loss(preds, y_input, selection=None):
    if selection is None:
        y = y_input
        output = preds
    else:
        y = y_input[selection]
        output = preds[selection]
    return ((output - y)**2).mean()/2

def loss_profile(preds, y, num_bins=100):
    y_min = y.min()
    y_max = y.max()
    bin_width = (y_max-y_min)/num_bins
    y_vals = np.linspace(y_min, y_max, num_bins)
    errors = np.zeros_like(y_vals)
    for i, y_val in enumerate(y_vals):
        bin_indices = np.where(abs(y-y_val) <= bin_width/2)
        if bin_indices[0].shape[0] > 0:
            errors[i] = 2*mse_loss(preds, y, selection=bin_indices[0])#x_featured[:,selection[0]], y_data[selection])
        else:
            errors[i] = -1
    good_errors = np.where(errors > 0)
    plt.scatter(y_vals[good_errors], np.log10(100*np.sqrt(errors[good_errors])))
    plt.title("Profile of error rate across different depths")
    plt.xlabel("Depth of cone (m)")
    plt.ylabel("log10(Average absolute error in centimeters)")
    plt.show()

def train_model(order, neg_only, residues, lamda, show_graph=False, verbose=False):
    x_featured, expr, num_poly = build_features(order, neg_only, x_data) # = np.ones((num_poly, x_data.shape[0]))

    if verbose:
        print(expr)

    actual_num_features = num_poly
    if "lin" in residues:
        actual_num_features += 3
    if "quad" in residues:
        actual_num_features += 4

    coeffs = np.ones(actual_num_features)*1e-3
    if verbose:
        print("Total num features:", actual_num_features)

    def pred(coeff, x_input):
        output = coeff[:num_poly].dot(x_input[0])
        if "lin" in residues:
            output += coeff[num_poly:num_poly+3].dot(x_input[1])**-1
        if "quad" in residues and "lin" not in residues:
            output += coeff[num_poly:].dot(x_input[2])**-1
        if "quad" in residues and "lin" in residues:
            output += coeff[num_poly+3:].dot(x_input[2])**-1
        return output

    def loss(coeff):
        return mse_loss(pred(coeff, (x_featured, x_lin_residue, x_quad_residue)), y_data)

    def loss_opt(coeff):
        return loss(coeff) + lamda*coeff[1:].dot(coeff[1:])/2  # only apply L2 norm to non-constant terms

    def loss_grad(coeff):
        reg_grad = lamda*coeff
        reg_grad[0] = 0
        return x_featured.dot(x_featured.T.dot(coeff[:num_poly]) - y_data)/y_data.shape[0] + reg_grad

    if verbose:
        print("Start loss:", loss(coeffs))
    if not residues:
        res = minimize(loss_opt, coeffs, method="BFGS", jac=loss_grad, options={"disp": verbose})
    else:
        res = minimize(loss_opt, coeffs, method="BFGS", jac="2-point", options={"disp": verbose})
    final_loss = loss(res.x)
    neg_only_str = " (neg only)" if neg_only else ""
    res_str = " + " + "/".join(residues) + " res" if residues else ""
    print(f"Order {order}{neg_only_str}{res_str}, reg={lamda}: {final_loss:.4f}")
    if verbose:
        print("Resulting parameters:", res.x)

    # Loss profile
    if show_graph:
        final_preds = pred(res.x, (x_featured, x_lin_residue, x_quad_residue))
        #print(final_preds.mean(), "final preds mean")
        loss_profile(final_preds, y_data)
    return final_loss, res.x, pred


experiments = [[0, False, [], 0],
               [1, False, [], 0],
               [2, False, [], 0],
               [3, False, [], 0],
               [1, True, [], 0],
               [2, True, [], 0],
               [3, True, [], 0],
               [4, True, [], 0],
               [5, True, [], 0],
               [6, True, [], 0],
               [7, True, [], 0],
               [8, True, [], 0],
               [9, True, [], 0],
               [0, True, ["lin", "quad"], 0],
               [1, True, ["lin", "quad"], 0],
               [2, True, ["lin", "quad"], 0],
               [3, True, ["lin", "quad"], 0],
               [4, True, ["lin", "quad"], 0],
               [0, True, ["lin"], 0],
               [1, True, ["lin"], 0],
               [2, True, ["lin"], 0],
               [3, True, ["lin"], 0],
               [4, True, ["lin"], 0],
               [0, True, ["quad"], 0],
               [1, True, ["quad"], 0],
               [2, True, ["quad"], 0],
               [0, False, ["lin", "quad"], 0],
               [1, False, ["lin", "quad"], 0],
               [2, False, ["lin", "quad"], 0],
               [3, False, ["lin", "quad"], 0],
               [4, False, ["lin", "quad"], 0],
               [0, False, ["lin"], 0],
               [1, False, ["lin"], 0],
               [2, False, ["lin"], 0],
               [3, False, ["lin"], 0],
               [4, False, ["lin"], 0],
               [0, False, ["quad"], 0],
               [1, False, ["quad"], 0],
               [2, False, ["quad"], 0],
              ]
#for experiment in experiments:
#    train_model(*experiment)


# VERIFY MODEL LOSS
FINAL_ORDER = 7
FINAL_NEG_ONLY = True
FINAL_RESIDUE = []
FINAL_LAMBDA = 1e-8
final_loss, final_param, pred_func = train_model(FINAL_ORDER, FINAL_NEG_ONLY, FINAL_RESIDUE, FINAL_LAMBDA, show_graph=True, verbose=True)
with open("train/_annotations.coco.json", "r") as f:
    annotations = json.load(f)

files = {img["file_name"]: img["id"] for img in annotations["images"]}
files = sorted(files.items(), key=lambda kv: kv[0])

INCHES_TO_CM = 2.54
LEFT_DIST = 251
RIGHT_DIST = 180
TRACK_LENGTH = 24 * 12
TRACK_SEPARATION = 6

left_angle = (LEFT_DIST**2 + TRACK_LENGTH**2 - RIGHT_DIST**2)/(2*LEFT_DIST*TRACK_LENGTH)

depths = np.zeros((len(files)))
bboxes = []
# residue not supported here
for i,(_, img_id) in enumerate(files):
    x_pos = (i//3)*TRACK_SEPARATION  # //3 since we take 3 pictures at each location
    depths[i] = np.sqrt(LEFT_DIST**2 + x_pos**2 - 2*LEFT_DIST*x_pos*left_angle)*INCHES_TO_CM
    point = annotations["annotations"][img_id]["bbox"]  # format is (min_x, min_y, width, height)
    bbox = _extended_bounding_box((point[0], max_y - point[1], point[0]+point[2], max_y - point[1] + point[3]))  # need to flip the y-axis
    bboxes.append(bbox)
bboxes = np.array(bboxes)

test_x = build_features(FINAL_ORDER, FINAL_NEG_ONLY, bboxes)[0]
test_x_lin_res = build_lin_res(bboxes)
test_x_quad_res = build_quad_res(bboxes)
#print(test_x.shape, test_x_lin_res.shape, test_x_quad_res.shape)
test_preds = pred_func(final_param, (test_x, test_x_lin_res, test_x_quad_res))
#print(test_preds.mean(), "test preds mean")
#print("depths mean", depths.mean())
ratio = (depths[:50]/test_preds[:50]).mean()
#depths /= ratio  # calibration ratio, is fixed
test_preds *= ratio
#print("depths mean after", depths.mean())
#print("y data means", y_data.mean())
print("Ratio was", ratio)
print("Test error", mse_loss(test_preds, depths))
loss_profile(test_preds/100, depths/100, num_bins=10)

