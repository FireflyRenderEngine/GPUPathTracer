#include "pch.h"

#include "../../Scene/Scene.h"

#include <memory>

TEST(SceneConstructorTest, DefaultConstructor) 
{
    std::shared_ptr<Scene> scene = std::make_shared<Scene>();
    ASSERT_TRUE(scene.get());
}