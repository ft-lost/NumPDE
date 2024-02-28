#ifndef SYSTEM_CALL_H
#define SYSTEM_CALL_H

/** @file
 * @brief NPDE homework systemcall::execute call with error handling
 * @author Manuel Saladin
 * @date 28.2.24
 * @copyright Developed at ETH Zurich
 */

#include <cstdlib>
#include <exception>
#include <string>

// Global function call to enable systemcall::execute calls with exception
namespace systemcall {
    void execute(const std::string &command) {
        int sys_out = std::system(command.c_str());
        if (sys_out != 0) {
            throw std::runtime_error("Error during system call: " + command + ", return value: " + std::to_string(sys_out));
        }
    }
}

#endif  // SYSTEM_CALL_H