import tensorflow as tf


def SGD(grads, w, b, lr):
    m1 = grads[0]
    m2 = grads[1]
    w.assign_sub(lr * m1)
    b.assign_sub(lr * m2)
    return w, b


def SGD_M(grads, w, b, lr, belta, m_w, m_b):
    m_w = belta * m_w + (1 - belta) * grads[0]
    m_b = belta * m_b + (1 - belta) * grads[1]
    w.assign_sub(lr * m_w)
    b.assign_sub(lr * m_b)
    return w, b


def Adagrad(grads, w, b, lr, v_w, v_b):
    epsilon = 1e-1
    m_w = grads[0]
    m_b = grads[1]
    v_w += tf.square(grads[0])
    v_b += tf.square(grads[1])
    w.assign_sub(lr * m_w / tf.sqrt(v_w + epsilon))
    b.assign_sub(lr * m_b / tf.sqrt(v_b + epsilon))
    return w, b


def Adadelta(grads, belta, w, b, v_w, v_b, lr):
    m_w = grads[0]
    m_b = grads[1]
    v_w = belta * v_w + (1 - belta) * tf.square(m_w)
    v_b = belta * v_b + (1 - belta) * tf.square(m_b)
    w.assign_sub(lr * m_w/tf.sqrt(v_w))
    b.assign_sub(lr * m_b/tf.sqrt(v_b))
    return w, b, v_w, v_b


def Adam(global_step, grads, belta1, belta2, w, b, v_w, v_b, m_w, m_b, lr):
    m_w = belta1 * m_w + (1 - belta1) * grads[0]
    m_b = belta1 * m_b + (1 - belta1) * grads[1]
    v_w = belta2 * v_w + (1 - belta2) * tf.square(grads[0])
    v_b = belta2 * v_b + (1 - belta2) * tf.square(grads[1])
    mm_w = m_w/(1 - tf.pow(belta1, int(global_step)))
    mm_b = m_b/(1 - tf.pow(belta1, int(global_step)))
    vv_w = v_w/(1 - tf.pow(belta2, int(global_step)))
    vv_b = v_b/(1 - tf.pow(belta2, int(global_step)))
    w.assign_sub(lr * mm_w/tf.sqrt(vv_w))
    b.assign_sub(lr * mm_b/tf.sqrt(vv_b))
    return w, b, m_w, m_b, v_w, v_b
