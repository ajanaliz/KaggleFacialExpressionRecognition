import numpy as np
import matplotlib.pyplot as plt

from util import getData

# the inputs to this map are integers 0 to 6, and the labels are the values below
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def main():
    X, Y = getData(balance_ones=False)

    # in an infinite loop
    while True:
        for i in xrange(7):  # we're going to loop through each of the 7 emotions
            x, y = X[Y == i], Y[Y == i]  # choose all the datapoints that equal this emotion
            N = len(y)  # get the number of datapoints that are equal to this emotion
            j = np.random.choice(N)  # randomly select a datapoint
            plt.imshow(x[j].reshape(48, 48), cmap='gray') # plot that data point -> because the data resides in flat vectors, we have to reshape it to a 48x48 image, and we set the colormap to gray since its a grayscale image
            plt.title(label_map[y[j]]) # we plot the title of this image to be the label, which is one of the emotions
            plt.show()
        prompt = raw_input('Quit? Enter Y:\n') # prompt to quit or not.
        if prompt == 'Y':
            break


if __name__ == '__main__':
    main()
