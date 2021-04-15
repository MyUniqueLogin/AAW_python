kernel void arrange(global int * buffer)

{ 

    const size_t gid = get_global_id(0); 

    buffer[gid] = convert_int(gid);

}