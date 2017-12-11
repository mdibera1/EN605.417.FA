//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Dan Ginsburg, Timothy Mattson
// ISBN-10:   ??????????
// ISBN-13:   ?????????????
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/??????????
//            http://www.????????.com
//

// simple.cl
//
//    This is a simple example demonstrating buffers and sub-buffer usage

__kernel void average(__global * buffer)
{
	size_t id = get_global_id(0);
    buffer[id] = (buffer[4*id] + buffer[4*id+1] + buffer[4*id+2] + buffer[4*id+3]) / 4.0;
}
