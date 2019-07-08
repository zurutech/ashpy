#Copyright 2019 Zuru Tech HK Limited. All Rights Reserved.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import tensorflow as tf

from ashpy.models.convolutional.unet import UNet


def main():
    x = tf.ones((1, 512, 512, 3))
    u_net = UNet(
        input_res=512,
        min_res=4,
        kernel_size=4,
        initial_filters=64,
        filters_cap=512,
        channels=3,
    )
    y = u_net(x, training=True)
    print(y.shape)


if __name__ == "__main__":
    main()
