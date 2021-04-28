import cv2
import numpy as np
import pyopencl as cl
import time

cameraIP = 'http://192.168.1.30:4747/video'

def make_background(bg, imgIn, context, queue):
	"apply morphological operation to image using GPU"
	
	# (1) setup OpenCL
	#platforms = cl.get_platforms() # a platform corresponds to a driver (e.g. AMD)
	#platform = platforms[0] # take first platform
	#devices = platform.get_devices(cl.device_type.GPU) # get GPU devices of selected platform
	#device = devices[0] # take first GPU
	
	# (2) get shape of input image, allocate memory for output to which result can be copied to
	shape = imgIn.T.shape
	imgOut = np.empty_like(imgIn)	
	
	# (2) create image buffers which hold images for OpenCL
	imgInBuf = cl.Image(context, cl.mem_flags.READ_ONLY, cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.UNORM_INT8), shape=shape) # holds a gray-valued image of given shape
	imgOutBuf = cl.Image(context, cl.mem_flags.WRITE_ONLY, cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.UNORM_INT8), shape=shape) # placeholder for gray-valued image of given shape
	imgInBuf2 = cl.Image(context, cl.mem_flags.READ_ONLY, cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.UNORM_INT8), shape=shape)
    
	# (3) load and compile OpenCL program
	program = cl.Program(context, open('background_kernel.cl').read()).build()

	# (3) from OpenCL program, get kernel object and set arguments (input image, operation type, output image)
	kernel = cl.Kernel(program, 'backgroundKernel') # name of function according to kernel.py
	kernel.set_arg(0, imgInBuf) # input image buffer
	kernel.set_arg(1, imgInBuf2) # input image buffer
	kernel.set_arg(2, imgOutBuf) # output image buffer
	
	# (4) copy image to device, execute kernel, copy data back
	cl.enqueue_copy(queue, imgInBuf, bg, origin=(0, 0), region=shape, is_blocking=False) # copy image from CPU to GPU
	cl.enqueue_copy(queue, imgInBuf2, imgIn, origin=(0, 0), region=shape, is_blocking=False) # copy image from CPU to GPU
	cl.enqueue_nd_range_kernel(queue, kernel, shape, None) # execute kernel, work is distributed across shape[0]*shape[1] work-items (one work-item per pixel of the image)
	cl.enqueue_copy(queue, imgOut, imgOutBuf, origin=(0, 0), region=shape, is_blocking=True) # wait until finished copying resulting image back from GPU to CPU
	
	return imgOut

def main():
    context = cl.create_some_context() # put selected GPU into context object
    queue = cl.CommandQueue(context) # create command queue for selected GPU and context

    vid = cv2.VideoCapture(cameraIP)
    
    fRate = 0
    start_time = time.time()
    
    ret, frame = vid.read()
    
    # read image
    bg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    while(True):
        
        ret, frame = vid.read()
    
        # read image
        I = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                     
        bg = make_background(bg, I, context, queue)
        
        cv2.imshow("Camera frame",I)
        cv2.imshow("Background",bg)
        
        fRate += 1   
        time1 = time.time()
        
        if time1 - start_time > 5:
            print('FPS rate: {FPSrate:.2f}'.format(FPSrate = fRate/(time1 - start_time)))
            
            fRate = 0
            start_time = time.time()
             
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
        
        
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
    
    
    #cv2.imwrite('background.png', background)
	
	


if __name__ == '__main__':
	main()