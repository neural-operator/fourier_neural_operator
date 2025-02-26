import torch
from ..fourier_continuation import FCLegendre

def test_fourier_continuation_1d():
    
    batch_size = 3
    length_signal = 101
    add_pts = 40
    
    x = torch.linspace(0, 1, length_signal).repeat(batch_size,1)
    f = torch.sin(16 * x) - torch.cos(8 * x)
    
    Extension = FCLegendre(n=2, d=add_pts)
    f_extend_one_side = Extension(f, dim=1, one_sided=True)
    f_extend_both_sides = Extension(f, dim=1, one_sided=False)
    
    # Check shape
    assert f_extend_both_sides.shape[-1] == f_extend_one_side.shape[-1] == f.shape[-1] + add_pts
    # Check values of original signal
    torch.testing.assert_close(f, f_extend_one_side[...,:length_signal])
    torch.testing.assert_close(f, f_extend_both_sides[...,add_pts//2:-add_pts//2])
    
    # # Plot to verify
    # import matplotlib.pyplot as plt
    # x_extended_one_side = torch.linspace(0, 1.4, 141) 
    # x_extended_both_sides = torch.linspace(-0.2, 1.2, 141) 
    # # Add 0.5 and -0.5 for plotting purposes
    # f_extend_one_side = f_extend_one_side + 0.5
    # f_extend_both_sides = f_extend_both_sides - 0.5
    # plt.figure(figsize=(10, 6))
    # plt.plot(x[0], f[0], 'k', label='original')
    # plt.plot(x_extended_one_side, f_extend_one_side[0] , 'b',label='extended_one_side')
    # plt.plot(x_extended_both_sides, f_extend_both_sides[0] , 'g', label='extended_both_sides')
    # plt.plot([0, 0], [-2.5, 2.5], '-', color='gray', lw=1)  
    # plt.plot([1, 1], [-2.5, 2.5], '-', color='gray', lw=1)  
    # plt.plot([0, 1.4], [f_extend_one_side[0,0],f_extend_one_side[0,0]], '--', color='b', lw=0.5)  
    # plt.plot([-0.2, 1.2], [f_extend_both_sides[0,0],f_extend_both_sides[0,0]], '--', color='g', lw=0.5) 
    # plt.legend()
    # plt.savefig('fourier_extension.pdf')


def test_fourier_continuation_2d():
    
    batch_size = 3
    length_signal = 101
    add_pts = 40
    
    x = torch.linspace(0, 1, length_signal).view(1, length_signal, 1).repeat(batch_size, 1, length_signal)
    y = torch.linspace(0, 1, length_signal).view(1, 1, length_signal).repeat(batch_size, length_signal, 1)
    f = torch.sin(12 * x)  - torch.cos(14 * y) + 3*x*y

    Extension = FCLegendre(n=3, d=add_pts)
    f_extend_one_side = Extension(f, dim=2, one_sided=True)
    f_extend_both_sides = Extension(f, dim=2, one_sided=False)

    # Check shape
    assert f_extend_one_side.shape[-1] == f.shape[-1] + add_pts
    assert f_extend_one_side.shape[-2] == f.shape[-2] + add_pts
    assert f_extend_both_sides.shape[-1] == f.shape[-1] + add_pts
    assert f_extend_both_sides.shape[-2] == f.shape[-2] + add_pts
    # Check values of original signal
    torch.testing.assert_close(f, f_extend_both_sides[...,add_pts//2:-add_pts//2, add_pts//2:-add_pts//2])
    torch.testing.assert_close(f, f_extend_one_side[...,:length_signal, :length_signal])

    # # Plot to verify
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(figsize=(12,4), nrows=1, ncols=3)
    # axs[0].imshow(f[0])
    # axs[0].set_title(r"Original", fontsize=17)
    # axs[1].imshow(f_extend_one_side[0])
    # axs[1].plot([length_signal, length_signal], [0, length_signal], '-', color='k', lw=3)
    # axs[1].plot([0, length_signal], [length_signal, length_signal], '-', color='k', lw=3)
    # axs[1].set_title(r"Extended one side", fontsize=17)
    # axs[2].imshow(f_extend_both_sides[0])
    # axs[2].set_title(r"Extended both sides", fontsize=17)
    # axs[2].plot([add_pts//2, length_signal + add_pts//2], [add_pts//2, add_pts//2], '-', color='k', lw=3)
    # axs[2].plot([add_pts//2, add_pts//2], [add_pts//2, length_signal + add_pts//2], '-', color='k', lw=3)
    # axs[2].plot([add_pts//2, length_signal + add_pts//2], [length_signal + add_pts//2, length_signal + add_pts//2], '-', color='k', lw=3)
    # axs[2].plot([length_signal + add_pts//2, length_signal + add_pts//2], [add_pts//2, length_signal + add_pts//2], '-', color='k', lw=3)
    # for ax in axs.flat:
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    # plt.savefig('fourier_extension.pdf')

