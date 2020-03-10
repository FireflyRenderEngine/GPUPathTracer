#include "pch.h"
#include "../Scene.h"

TEST(TestingScene, Constructor) 
{
    Scene* scene = new Scene();
    ASSERT_TRUE(scene);
}