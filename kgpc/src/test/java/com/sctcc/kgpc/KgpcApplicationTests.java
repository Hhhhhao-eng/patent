package com.sctcc.kgpc;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.ResponseEntity;
import org.springframework.web.client.RestTemplate;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.when;
import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
class PatentControllerTest {
    
    @Autowired
    private PatentController controller;
    
    @MockBean
    private RestTemplate restTemplate;
    
    @Test
    void testRecommendationSuccess() {
        // Mock Django响应
        when(restTemplate.exchange(
                anyString(), 
                eq(HttpMethod.GET), 
                isNull(), 
                any(ParameterizedTypeReference.class))
            .thenReturn(ResponseEntity.ok(Map.of("obj", List.of()))));
        
        ResponseEntity<?> response = controller.getPatentRecommendations("CN100001", 5);
        assertEquals(HttpStatus.OK, response.getStatusCode());
    }
}
