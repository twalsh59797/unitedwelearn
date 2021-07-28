from PIL import Image

im1 = Image.open('000000002473_3.jpg')
im2 = Image.open('000000002473.jpg')


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (int(im1.width/2) + im2.width, im1.height))
    im1 = im1.crop((0, 0, im1.width/2, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

concat_img = get_concat_h(im1, im2)
part1 = concat_img.crop((0+200, 0, concat_img.width/3, concat_img.height))
part1.save('concat1_part1.png')
part2 = concat_img.crop((concat_img.width/3+200, 0, 2*concat_img.width/3, concat_img.height))
part2.save('concat1_part2.png')
part3 = concat_img.crop((2*concat_img.width/3+200, 0, concat_img.width, concat_img.height))
part3.save('concat1_part3.png')

im1 = Image.open('000000001425_3.jpg')
im2 = Image.open('000000001425.jpg')
concat_img = get_concat_h(im1, im2)
part1 = concat_img.crop((0+30, 0, concat_img.width/3-70, concat_img.height))
part1.save('concat2_part1.png')
part2 = concat_img.crop((concat_img.width/3+30, 0, 2*concat_img.width/3-70, concat_img.height))
part2.save('concat2_part2.png')
part3 = concat_img.crop((2*concat_img.width/3+30, 0, concat_img.width-70, concat_img.height))
part3.save('concat2_part3.png')


im1 = Image.open('000000027982_2.jpg')
im2 = Image.open('000000027982.jpg')
concat_img = get_concat_h(im1, im2)
part1 = concat_img.crop((0+160, 0, concat_img.width/3, concat_img.height))
part1.save('concat3_part1.png')
part2 = concat_img.crop((concat_img.width/3+160, 0, 2*concat_img.width/3, concat_img.height))
part2.save('concat3_part2.png')
part3 = concat_img.crop((2*concat_img.width/3+160, 0, concat_img.width, concat_img.height))
part3.save('concat3_part3.png')


