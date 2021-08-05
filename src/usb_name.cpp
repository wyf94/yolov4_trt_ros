#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <vector>
#include <cstring>
#include <assert.h>

std::vector<std::string> split(const std::string& str, const std::string& delim) {
	std::vector<std::string> res;
	if("" == str) return res;
	char * strs = new char[str.length() + 1] ; 
	strcpy(strs, str.c_str());

	char * d = new char[delim.length() + 1];
	strcpy(d, delim.c_str());

	char *p = strtok(strs, d);
	while(p) {
		std::string s = p; 
		res.push_back(s); 
		p = strtok(NULL, d);
	}

	return res;
}

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;

}

std::string process_result(const std::string& result,const std::string& id_vendor,const std::string& id_product) {
    std::vector<std::string> usbs = split(result,"\n");
    bool product = false;
    bool vendor = false;
    std::string ret_dev="";
    for(int i =0;i<usbs.size();i++) {
            std::string query_cmd = "udevadm info --query=all "+usbs[i]+" | grep 'VENDOR_ID\\|MODEL_ID'";
	    std::string query_return = exec(query_cmd.c_str());
	    
	    std::vector<std::string> query_result = split(query_return,"\n");
	    for( int j =0;j<query_result.size();j++) {
		    std::string temp = split(query_result[j],"=")[1];
		    if(temp == id_vendor) {
		    vendor = true;
		    }
		    if(temp == id_product) {
		    product = true;
		    }
		    if(product && vendor) {
		    return usbs[i];
		    }
	    }
    }
    return ret_dev;
}
