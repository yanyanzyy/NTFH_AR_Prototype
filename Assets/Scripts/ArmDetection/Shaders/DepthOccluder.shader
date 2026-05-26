Shader "ArmDetection/DepthOccluder"
{
    // Renders geometry to the depth buffer only (ColorMask 0, no colour output).
    // Queue = Geometry-10 so it executes before the ArmOverlayUnlit quad.
    // Placing this shader on a sphere at the wearer's tracked wrist position causes the
    // overlay to fail ZTest LEqual at those pixels --> wearer's arm appears in front of the overlay.
    SubShader
    {
        Tags
        {
            "RenderType"      = "Opaque"
            "Queue"           = "Geometry-10"
            "RenderPipeline"  = "UniversalPipeline"
        }
        LOD 100

        Pass
        {
            Name "DepthOccluder"
            ZWrite On
            ZTest  LEqual
            ColorMask 0
            Cull Back

            HLSLPROGRAM
            #pragma vertex   vert
            #pragma fragment frag
            #pragma target 2.0
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                UNITY_SETUP_INSTANCE_ID(IN);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);
                OUT.positionHCS = TransformObjectToHClip(IN.positionOS.xyz);
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                return 0; // colour is masked; this line is never actually written
            }
            ENDHLSL
        }
    }
}
