#pragma once

enum class MATERIAL_BXDF { GLASS_FRESNEL = 0, DIFFUSE, MICROFACET, SUBSURFACE, METAL, EMITTER };

class Material
{
public:
	void InitializeMaterialBXDFType(MATERIAL_BXDF type) {
		m_materialBXDF = type;
	}
private:
	MATERIAL_BXDF m_materialBXDF;
};