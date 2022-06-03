"""
SLIC Pipeline for DIP Project
"""

import math
from skimage import io, color
import numpy as np


def create_cluster(image, h, w):
    """
    function for creating cluster
    :param image:
    :param h:
    :param w:
    :return:
    """
    h = int(h)
    w = int(w)
    return SuperPixel(h, w,
                      image[h][w][0],
                      image[h][w][1],
                      image[h][w][2])


class SuperPixel(object):
    """
    Super Pixel class for defining a super pixel
    """
    cluster_index = 1

    def __init__(self, h, w, l=0, a=0, b=0):
        self._h = h
        self._w = w
        self._l = l
        self._a = a
        self._b = b
        SuperPixel.cluster_index += 1
        self.pixels = []
        self.no = self.cluster_index

    def change(self, h, w, l, a, b):
        """
        updating the values
        :param h:
        :param w:
        :param l:
        :param a:
        :param b:
        """
        self._h = h
        self._w = w
        self._l = l
        self._a = a
        self._b = b

    def __str__(self):
        return str(self._h) + ":" + str(self._w) + ":" + str(self._l) + ":" + str(self._a) + ":" + str(self._b)

    def __repr__(self):
        return self.__str__()


class SLICProcessor(object):
    """
    Main class for functioning
    """

    def __init__(self, image, k, m, mode="slicx"):
        self.mode = mode
        self.K = k
        self.M = m

        self.image = image
        self.total_height = self.image.shape[0]
        self.total_width = self.image.shape[1]
        self.N = self.total_height * self.total_width
        self.S = int(math.sqrt(self.N / self.K))

        self.super_pixel = []
        self.label = {}
        self.dis = np.full((self.total_height, self.total_width), np.inf)

    def get_gradient(self, h, w):
        """
        Quantify the pixel position
        :param h:
        :param w:
        :return:
        """
        if w + 1 >= self.total_width:
            w = self.total_width - 2
        if h + 1 >= self.total_height:
            h = self.total_height - 2

        gradient = self.image[h + 1][w + 1][0] - self.image[h][w][0] + \
                   self.image[h + 1][w + 1][1] - self.image[h][w][1] + \
                   self.image[h + 1][w + 1][2] - self.image[h][w][2]
        return gradient

    def iterations(self, iter_count=10):

        # Forming the Even super_pixel
        h = self.S / 2
        w = self.S / 2
        while h < self.total_height:
            while w < self.total_width:
                cluster = create_cluster(self.image, h, w)
                # Checking if any cluster center in on the gradient and moving it on either side
                cluster_gradient = self.get_gradient(cluster._h, cluster._w)
                for dh in range(-1, 2):
                    for dw in range(-1, 2):
                        _h = cluster._h + dh
                        _w = cluster._w + dw
                        new_gradient = self.get_gradient(_h, _w)
                        if new_gradient < cluster_gradient:
                            cluster.change(_h, _w, self.image[_h][_w][0], self.image[_h][_w][1], self.image[_h][_w][2])
                            cluster_gradient = new_gradient
                self.super_pixel.append(cluster)
                w += self.S
            w = self.S / 2
            h += self.S

        for i in range(iter_count):
            for cluster in self.super_pixel:
                for h in range(cluster._h - 2 * self.S, cluster._h + 2 * self.S):
                    if h < 0 or h >= self.total_height:
                        continue
                    for w in range(cluster._w - 2 * self.S, cluster._w + 2 * self.S):
                        if w < 0 or w >= self.total_width:
                            continue
                        L, A, B = self.image[h][w]
                        Dc = math.sqrt(
                            math.pow(L - cluster._l, 2) +
                            math.pow(A - cluster._a, 2) +
                            math.pow(B - cluster._b, 2))
                        Ds = math.sqrt(
                            math.pow(h - cluster._h, 2) +
                            math.pow(w - cluster._w, 2))
                        D = math.sqrt(math.pow(Dc / self.M, 2) + math.pow(Ds / self.S, 2))
                        if D < self.dis[h][w]:
                            if (h, w) not in self.label:
                                self.label[(h, w)] = cluster
                                cluster.pixels.append((h, w))
                            else:
                                self.label[(h, w)].pixels.remove((h, w))
                                self.label[(h, w)] = cluster
                                cluster.pixels.append((h, w))
                            self.dis[h][w] = D
                if self.mode == "slic":
                    sum_h = sum_w = number = 0
                    for p in cluster.pixels:
                        sum_h += p[0]
                        sum_w += p[1]
                        number += 1
                    _h = int(sum_h / number)
                    _w = int(sum_w / number)
                    cluster.change(_h, _w, self.image[_h][_w][0], self.image[_h][_w][1], self.image[_h][_w][2])
                else:
                    sum_h = sum_w = number = 0
                    _l_temp = 0
                    _a_temp = 0
                    _b_temp = 0
                    radius = int((math.sqrt(self.total_height * self.total_width / (2 * self.K)) - 1) // 2)
                    area = int((radius * 2 + 1) ** 2)

                    for p in cluster.pixels:
                        sum_h += p[0]
                        sum_w += p[1]
                        number += 1
                    _h = int(sum_h / number)
                    _w = int(sum_w / number)
                    count = 0
                    for y in range(_h - radius, _h + radius):
                        if y < 0 or y >= self.total_height:
                            continue
                        for x in range(_w - radius, _w + radius):
                            if x < 0 or x >= self.total_width:
                                continue
                            count += 1
                            _l_temp += self.image[y][x][0]
                            _a_temp += self.image[y][x][1]
                            _b_temp += self.image[y][x][2]

                    _l_temp = int(_l_temp / count)
                    _a_temp = int(_a_temp / count)
                    _b_temp = int(_b_temp / count)
                    cluster.change(_h, _w, _l_temp, _a_temp, _b_temp)


        image_arr = np.copy(self.image)
        for cluster in self.super_pixel:
            for p in cluster.pixels:
                image_arr[p[0]][p[1]][0] = cluster._l
                image_arr[p[0]][p[1]][1] = cluster._a
                image_arr[p[0]][p[1]][2] = cluster._b
            image_arr[cluster._h][cluster._w][0] = 0
            image_arr[cluster._h][cluster._w][1] = 0
            image_arr[cluster._h][cluster._w][2] = 0
        return image_arr


if __name__ == '__main__':
    # p = SLICProcessor('Lenna.png', 1000, 5)
    # p.iterate_10times()
    rgb = io.imread("Lenna.png")
    lab_arr = color.rgb2lab(rgb)
    mode = "slic"
    p = SLICProcessor(lab_arr, 1000, 40, "slic")
    lab_arr = p.iterations()
    if mode == "slic":
        name = 'lenna_slic.png'
    else:
        name = 'lenna_slic_centroid_x.png'
    rgb_arr = color.lab2rgb(lab_arr)
    io.imsave(name, rgb_arr)
