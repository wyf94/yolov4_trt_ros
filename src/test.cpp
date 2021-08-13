#include <test.h>

void test::Func()
{
    str = "Test CPP";
    std::cout<<str<<std::endl;

    YAML::Node yaml2;

    yaml2 = YAML::LoadFile("/root/ros2_ws/src/BITCQ_Camera/config/Galaxy_Camera/galaxy_camera.yaml");

    int q = yaml2["settings"][1]["width"].as<int>();
    std::cout << q <<std::endl;

    std::cout << "Main OpenCV version : " << CV_VERSION << std::endl;

    s.Func();
}