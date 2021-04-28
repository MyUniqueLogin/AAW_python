// (4b) settings of sampler to read pixel values from image: 
// * coordinates are pixel-coordinates
// * no interpolation between pixels
// * pixel values from outside of image are taken from edge instead
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;





// (1) kernel with 3 arguments: input and output image, and operation (dilation=0 or erosion=1)
__kernel void backgroundKernel(__read_only image2d_t bg, __read_only image2d_t in, __write_only image2d_t out)
{
    //float pix_in = 0.0f;
    float pix = 0.0f;
    
	// (2) IDs of work-item represent x y and z coordinates in image
	const int x = get_global_id(0);
	const int y = get_global_id(1);

    const float pix_in = read_imagef(in, sampler, (int2)(x, y)).s0;
    const float pix_bg = read_imagef(bg, sampler, (int2)(x, y)).s0;
    float alfa = 0.01f;
    pix = pix_bg*(1-alfa) + pix_in*alfa;
	// (6) write value of pixel to output image at location (x, y)
	write_imagef(out, (int2)(x, y), (float4)(pix, 0.0f, 0.0f, 0.0f));
}