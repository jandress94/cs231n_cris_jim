import numpy as np

label_thresholds = np.zeros((17,))

def transform_target_to_1_0_vect(target):
    vect = np.zeros((17,))
    vect[target] = 1
    return vect

def print_progress(index, collection_len, prompt, print_every = 10, loss = None):
    if index % print_every == 0:
        total = collection_len if isinstance(collection_len, int) else len(collection_len)
        if loss is None: print('%s %d / %d' % (prompt, index, total))
        else: print('%s %d / %d   loss: %f' % (prompt, index, total, loss))

def compute_f2(scores, y, threshold, axis, eps = 1e-8):
  preds = scores >= threshold
  preds = preds.cpu().int()
  y = y.int()
    
  pred_pos = torch.sum(preds, axis).float()
  real_pos = torch.sum(y, axis).float()
  true_pos = torch.sum(y * preds, axis).float()
  
  p = 1.0 * true_pos / (pred_pos + eps)
  r = 1.0 * true_pos / (real_pos + eps)

  beta = 2
  return (1.0 + beta**2)*p*r / (beta**2 * p + r + eps)

def recompute_thresholds(model, loader, dtype, eps = 1e-8):
  scores_list = []
  ys = []

  for i, (x, y) in enumerate(loader):
    print_progress(i, loader, 'getting scores to compute thresholds')
    x_var = Variable(x.type(dtype), volatile = True)
    scores = model(x_var)
    normalized_scores = torch.sigmoid(scores)
    scores_list.append(normalized_scores)
    ys.append(y)

  scores = torch.cat(scores_list, 0).data
  ys = torch.cat(ys, 0)

  best_thresh = np.zeros((17,))
  best_f2 = -np.ones((17,))

  for t in range(1000):
    print_progress(t, 1000, 'Recomputing thresholds')
    thresh = (1 + t) * 0.001

    f2 = compute_f2(scores, ys, thresh, 0).numpy()

    better_mask = f2 > best_f2
    better_mask = better_mask.astype(np.int)
    best_thresh = (1 - better_mask) * best_thresh + better_mask * thresh
    best_f2 = (1 - better_mask) * best_f2 + better_mask * f2

  return best_thresh

def check_f2(model, loader, dtype, recomp_thresh = False, eps = 1e-8):
  """
  Check the accuracy of the model.
  """
  global label_thresholds
  # Set the model to eval mode
  model.eval()

  if recomp_thresh: 
    label_thresholds = recompute_thresholds(model, loader, dtype)
    print('Computed new thresholds:', label_thresholds)

  running_f2, num_samples = 0.0, 0

  #thresholds = torch.Tensor([0.2625, 0.2375, 0.245, 0.21, 0.205, 0.1625, 0.265, 0.2175, 0.1925, 0.12, 0.2225, 0.14, 0.1375, 0.19, 0.085, 0.0475, 0.0875]).type(dtype)
  thresholds = torch.Tensor(label_thresholds).type(dtype)

  for mini_index, (x, y) in enumerate(loader):
    if thresholds.size() != y.size():
      thresholds = torch.Tensor(label_thresholds).type(dtype)
      thresholds = torch.cat([thresholds for _ in range(x.size(0))], 0)

    print_progress(mini_index, loader, 'Evaluating minibatch')
    # Cast the image data to the correct type and wrap it in a Variable. At
    # test-time when we do not need to compute gradients, marking the Variable
    # as volatile can reduce memory usage and slightly improve speed.
    x_var = Variable(x.type(dtype), volatile=True)

    scores = model(x_var)
    normalized_scores = torch.sigmoid(scores)

    f2 = compute_f2(normalized_scores.data, y, thresholds, 1)
    running_f2 += torch.sum(f2)
    num_samples += x.size(0)

  f2 = running_f2 / num_samples
  return f2