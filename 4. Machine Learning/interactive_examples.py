import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

top = cm.get_cmap('Oranges_r', 128)
bottom = cm.get_cmap('Blues', 128)

newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
newcmp = ListedColormap(newcolors, name='OrangeBlue')


x = np.linspace(-5, 5)
y = np.random.normal(.6*x+1)
    
N = 150
class_1_x = np.random.normal(2, 1.7, N)
class_1_y = np.random.normal(-1, 1.7, N)
class_2_x = np.random.normal(-1, 1.7, N)
class_2_y = np.random.normal(2, 1.7, N)
class_1 = (class_1_x, class_1_y)
class_2 = (class_2_x, class_2_y)


alpha = 2 * np.pi * np.random.uniform(size=N)
r = np.random.normal(3, .5, N)

class_inner_x = np.random.normal(0, 1, N)
class_inner_y = np.random.normal(0, 1, N)
x_outer = np.linspace(-2, 2, N)
class_outer_x = r*np.cos(alpha)
class_outer_y = r*np.sin(alpha)
class_inner = (class_inner_x, class_inner_y)
class_outer = (class_outer_x, class_outer_y)

    

def get_m_b_sliders():
    params = {
        'value':0, 'min':-3, 'max':3, 'step':0.05, 
        'continuous_update':False, 'orientation':'horizontal' 
    }
    m = widgets.FloatSlider(**params, description='slope m:')
    b = widgets.FloatSlider(**params, description='intercept b:')
    return m, b

    
def linear_example():
    m, b = get_m_b_sliders()
    
    out = widgets.interactive_output(lambda m, b: plot_it(x, y, linear, m, b), {'m': m, 'b': b})
    display(widgets.VBox([out, widgets.VBox([m, b])], 
                         layout=widgets.Layout(align_items='center')
                        )
           )
    
def linear_classification():
    m, b = get_m_b_sliders()
        
    out = widgets.interactive_output(
        lambda m, b: plot_classes(class_1, class_2, linear, m, b), {'m': m, 'b': b}
    )
    display(widgets.VBox([out, widgets.VBox([m, b])], 
                         layout=widgets.Layout(align_items='center')
                        )
           )
    
def linear_classification_accuracy():
    m, b = get_m_b_sliders()
        
    out = widgets.interactive_output(
        lambda m, b: plot_classes(class_1, class_2, linear, m, b), {'m': m, 'b': b}
    )
    loss_text = widgets.interactive_output(
        lambda m, b: display(
            widgets.FloatText(
            value=round(np.mean([
                1-accuracy(class_1[1], linear(class_1[0], m, b)), 
                accuracy(class_2[1], linear(class_2[0], m, b))
            ]),3),
            description='accuracy:',
            disabled=False, 
            layout={'width': '100%'}
            )
        ),
        {'m': m, 'b': b}
    )
    display(widgets.VBox([out, widgets.VBox([m, b, loss_text])], 
                         layout=widgets.Layout(align_items='center')
                        )
           )
    
def linear_classification_accuracy_and_bce():
    m, b = get_m_b_sliders()
        
    out = widgets.interactive_output(
        lambda m, b: plot_proba(class_1, class_2, linear, m, b), {'m': m, 'b': b}
    )
    loss_plot(None, None, reset=True)
    loss_display = widgets.interactive_output(
        lambda m, b: loss_plot(np.mean(
            [binary_crossentropy(0, linear_sigmoid(class_1_x, class_1_y, m, b)),
             binary_crossentropy(1, linear_sigmoid(class_2_x, class_2_y, m, b))
            ]), loss_name='b. c. e.'
                              ),
        {'m': m, 'b': b}
    )
    
    acc_text = widgets.interactive_output(
        lambda m, b: display(
            widgets.FloatText(
            value=round(np.mean([
                1-accuracy(class_1[1], linear(class_1[0], m, b)), 
                accuracy(class_2[1], linear(class_2[0], m, b))
            ]),3),
            description='accuracy:',
            disabled=False, 
            layout={'width': '100%'}
            )
        ),
        {'m': m, 'b': b}
    )
    bce_text = widgets.interactive_output(
        lambda m, b: display(
            widgets.FloatText(
            value=round(np.mean(
                [binary_crossentropy(0, linear_sigmoid(class_1_x, class_1_y, m, b)),
                 binary_crossentropy(1, linear_sigmoid(class_2_x, class_2_y, m, b))
                ])
                ,3),
            description='b. c. e. loss:',
            disabled=False, 
            layout={'width': '100%'}
            )
        ),
        {'m': m, 'b': b}
    )
    display(widgets.VBox([widgets.HBox([out, loss_display]), widgets.VBox([m, b, acc_text, bce_text])], 
                         layout=widgets.Layout(align_items='center')
                        )
           )

def circle_and_the_line():
    m, b = get_m_b_sliders()
        
    out = widgets.interactive_output(
        lambda m, b: plot_proba(class_inner, class_outer, linear, m, b), {'m': m, 'b': b}
    )
    loss_plot(None, None, reset=True)
    loss_display = widgets.interactive_output(
        lambda m, b: loss_plot(np.mean(
            [binary_crossentropy(0, linear_sigmoid(class_inner_x, class_inner_y, m, b)),
             binary_crossentropy(1, linear_sigmoid(class_outer_x, class_outer_y, m, b))
            ]), loss_name='b. c. e.'
                              ),
        {'m': m, 'b': b}
    )
    
    acc_text = widgets.interactive_output(
        lambda m, b: display(
            widgets.FloatText(
            value=round(np.mean([
                1-accuracy(class_inner[1], linear(class_inner[0], m, b)), 
                accuracy(class_outer[1], linear(class_outer[0], m, b))
            ]),3),
            description='accuracy:',
            disabled=False, 
            layout={'width': '100%'}
            )
        ),
        {'m': m, 'b': b}
    )
    bce_text = widgets.interactive_output(
        lambda m, b: display(
            widgets.FloatText(
            value=round(np.mean(
                [binary_crossentropy(0, linear_sigmoid(class_inner_x, class_inner_y, m, b)),
                 binary_crossentropy(1, linear_sigmoid(class_outer_x, class_outer_y, m, b))
                ])
                ,3),
            description='b. c. e. loss:',
            disabled=False, 
            layout={'width': '100%'}
            )
        ),
        {'m': m, 'b': b}
    )
    display(widgets.VBox([widgets.HBox([out, loss_display]), widgets.VBox([m, b, acc_text, bce_text])], 
                         layout=widgets.Layout(align_items='center')
                        )
           )

def simple_network(x, y, w01, w02, w11, w12, w21, w22, wx0, wx1, wx2):
    first_intermediate = sigmoid(w01+w11*x+w21*y)
    second_intermediate = sigmoid(w02+w12*x+w22*y)
    return sigmoid(wx0 + wx1*first_intermediate + wx2*second_intermediate)

def sigmoid(t):
    return 1/(1+np.exp(-t))

def get_N_sliders(labels, values=None):
    params = {
        'min':-5, 'max':5, 'step':0.02, 
        'continuous_update':False, 'orientation':'horizontal' 
    }
    if values is None:
        values = np.zeros(len(labels))
    return [widgets.FloatSlider(**params, value=val, description=label) for val, label in zip(values, labels)]
 

def circle_with_hidden_neurons():
    labels = [
        '$w_{0, 1}$', '$w_{0, 2}$', 
        '$w_{1, 1}$', '$w_{1, 2}$',  
        '$w_{2, 1}$', '$w_{2, 2}$',  
        '${w^\prime}_0$', '${w^\prime}_1$','${w^\prime}_2$'
    ]
    values = [-3.46, 1.64, 1.66, -0.94, 2.28, -1.68, 1.38, 4.04, -3.32]
    sliders = get_N_sliders(labels, values)

    out = widgets.interactive_output(
        lambda  w01, w02, w11, w12, w21, w22, wx0, wx1, wx2: plot_proba_nn(class_inner, class_outer, 
        simple_network, w01, w02, w11, w12, w21, w22, wx0, wx1, wx2), 
        {k:v for k, v in zip(
            ['w01', 'w02', 'w11', 'w12', 'w21', 'w22', 'wx0', 'wx1', 'wx2'],
            sliders
        )}
    )
    
    display(widgets.HBox([out, widgets.VBox(
            sliders, 
            layout=widgets.Layout(align_items='center')
            )
        ]), 
        layout=widgets.Layout(align_items='center')
    )
        

            
def linear_sigmoid(x, y, m, b):
    distance = -(m*x - y + b)/np.sqrt(m**2+1)
    return 1/(1+np.exp(-distance))
    
def binary_crossentropy(y_true, p):
    loss = -np.mean(y_true*np.log(p) + (1-y_true)*np.log(1-p))
    return loss
    
def linear_example_with_loss():
    m, b = get_m_b_sliders()
        
    out = widgets.interactive_output(lambda m, b: plot_it(x, y, linear, m, b), {'m': m, 'b': b})
    loss_text = widgets.interactive_output(
        lambda m, b: display(
            widgets.FloatText(
            value=round(mse(y, linear(x, m, b)), 3),
            description='m. s. e. loss:',
            disabled=False, 
            layout={'width': '100%'}
            )
        ),
        {'m': m, 'b': b}
    )
    loss_plot(None, None, reset=True)
    loss_display = widgets.interactive_output(lambda m, b: loss_plot(mse(y, linear(x, m, b))), 
                                              {'m': m, 'b': b}
                                             )
    display(widgets.VBox([widgets.HBox([out, loss_display]), widgets.VBox([m, b, loss_text])],
                        layout=widgets.Layout(align_items='center')
                        )
           )
    
def linear_example_gradient_descent():
    m, b = get_m_b_sliders()
    
    out = widgets.interactive_output(lambda m, b: plot_it(x, y, linear, m, b), {'m': m, 'b': b})
    loss_text = widgets.interactive_output(
        lambda m, b: display(
            widgets.FloatText(
            value=round(mse(y, linear(x, m, b)), 3),
            description='m. s. e. loss:',
            disabled=True,
            layout={'width': '100%'}
            )
        ),
        {'m': m, 'b': b}
    )
    loss_plot(None, None, reset=True)
    loss_display = widgets.interactive_output(lambda m, b: loss_plot(mse(y, linear(x, m, b))), 
                                              {'m': m, 'b': b}
                                             )
    
    lr = widgets.FloatText(
        value=1e-2,
        step=1e-3,
        description='learning rate:',
        disabled=False,
        layout={'width': '40%'}
    )
    do_step_button = widgets.Button(description='Descend!')
    do_step_button.on_click(lambda _: gradient_descent_step(x, y, m, b, lr))
    
    display(
        widgets.VBox([
            widgets.HBox([out, loss_display]), 
            widgets.HBox([
                widgets.VBox([m, b, widgets.HBox([lr, do_step_button]), loss_text])
            ])
        ],layout=widgets.Layout(align_items='center')
        )
    )
    

def gradient_descent_step(x, y, m_w, b_w, lr):
    grads = ( 2*np.mean( (y - (x*m_w.value + b_w.value))*(-x)), 
              2*np.mean( (-1)*(y - (x*m_w.value + b_w.value) ) )
            )
    m_w_new = m_w.value - lr.value*grads[0]
    b_w_new = b_w.value - lr.value*grads[1]
    m_w.value, b_w.value = m_w_new, b_w_new
    return 
    
def plot_it(x, y, f, m, b):
    plt.plot(x, y, '.', label='Measurement data')
    plt.plot(x, f(x, m, b), '--', label='Manual fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')
    
def step_function(x, y, m, b):
    return np.where(m*x+b > y, 1, 0)

def plot_classes(class_1, class_2, f, m, b):
    plt.plot(class_1[0], class_1[1], 'o', color='C1', label='Class 1')
    plt.plot(class_2[0], class_2[1], 's', color='C0', label='Class 2')
    xlims = plt.xlim()
    ylims = plt.ylim()
    x = np.linspace(xlims[0], xlims[1])
    y = np.linspace(ylims[0], ylims[1])
    
    X, Y = np.meshgrid(x, y)
    Z = np.where(Y > X*m+b, 1, 0)
    newcmp = ListedColormap(newcolors)
    plt.pcolormesh(X, Y, Z, cmap=newcmp, vmin=-1, vmax=2)

    plt.plot(x, f(x, m, b), '--', color='k', label='Decision boundary')
    
    plt.xlim(*xlims)
    plt.ylim(*ylims)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=2, bbox_to_anchor=(1, 1))
    
def plot_proba(class_1, class_2, f, m, b):
    plt.plot(class_1[0], class_1[1], 'o', color='C1', label='Class 1')
    plt.plot(class_2[0], class_2[1], 's', color='C0', label='Class 2')
    xlims = plt.xlim()
    ylims = plt.ylim()
    x = np.linspace(xlims[0], xlims[1])
    y = np.linspace(ylims[0], ylims[1])
    
    X, Y = np.meshgrid(x, y)
    Z = linear_sigmoid(X, Y, m, b)
    newcmp = ListedColormap(newcolors)
    plt.pcolormesh(X, Y, Z, cmap=newcmp, vmin=-1, vmax=2)

    plt.plot(x, f(x, m, b), '--', color='k', label='Decision boundary')
    
    plt.xlim(*xlims)
    plt.ylim(*ylims)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=2, bbox_to_anchor=(1, 1))

def plot_proba_nn(class_1, class_2, f, *args):
    plt.plot(class_1[0], class_1[1], 'o', color='C1', label='Class 1')
    plt.plot(class_2[0], class_2[1], 's', color='C0', label='Class 2')
    xlims = plt.xlim()
    ylims = plt.ylim()
    x = np.linspace(xlims[0], xlims[1])
    y = np.linspace(ylims[0], ylims[1])
    
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y, *args)
    Z = np.where(Z > .5, 1, 0)
    newcmp = ListedColormap(newcolors)
    plt.pcolormesh(X, Y, Z, cmap=newcmp, vmin=-1, vmax=2)

    # plt.plot(x, f(x, m, b), '--', color='k', label='Decision boundary')
    
    plt.xlim(*xlims)
    plt.ylim(*ylims)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=2, bbox_to_anchor=(1, 1))
    
def loss_plot(loss, loss_name='m. s. e.', reset=False):
    if reset:
        loss_plot.history = {'x':[], 'loss':[]}
        return    
    
    if not hasattr(loss_plot, 'history'):
        loss_plot.history = {'x':[], 'loss':[]}
    
    #loss = loss_func(y_true, y_pred)
    loss_plot.history['x'].append(len(loss_plot.history['x']))
    loss_plot.history['loss'].append(loss)
    
    plt.plot(loss_plot.history['x'], loss_plot.history['loss'], 'o-', label='loss')
    plt.xlabel('iteration')
    plt.ylabel(f'{loss_name} loss')

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc='best')

def linear(x, m, b):
    return m*x + b

def mse(y_true, y_pred):
    return np.mean( (y_true-y_pred)**2 )

def accuracy(y_true, db):
    return np.mean(np.where(y_true > db, 1, 0))