#include "../Scene.h"
#include "pch.h"

TEST(TestingScene, Constructor) 
{
    Scene* scene = new Scene();
    ASSERT_TRUE(scene);
}