# define executable
# NOTE: ode45.h is in the global include path because homeworks also use it,
# so we don't need to specify it here
set(ode45sources ode45test.cc)
add_executable(lecturecodes.ode45test ${ode45sources})
target_link_libraries(lecturecodes.ode45test PUBLIC Eigen3::Eigen)

set(ode45stiffsources ode45stiff.cc)
add_executable(lecturecodes.ode45stiff ${ode45stiffsources})
target_link_libraries(lecturecodes.ode45stiff PUBLIC Eigen3::Eigen)

set(odeintssctrlsources odeintssctrltest.cc odeintssctrl.h)
add_executable(lecturecodes.odeintssctrltest ${odeintssctrlsources})
target_link_libraries(lecturecodes.odeintssctrltest PUBLIC Eigen3::Eigen)

set(embeddedrkssm embeddedrkssm.cc)
add_executable(lecturecodes.embeddedrkssm ${embeddedrkssm})
target_link_libraries(lecturecodes.embeddedrkssm PUBLIC Eigen3::Eigen)

#set(odeintadaptsources odeintadapttest.cc odeintadapt.h)
#add_executable(lecturecodes.odeintadapt ${odeintadaptsources})
#target_link_libraries(lecturecodes.odeintadapt PUBLIC Eigen3::Eigen)
